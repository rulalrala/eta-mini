//! api/app.py 와 같은 HTTP 계약을 구현한다.
//!   GET  /health → {"status": "ok", "model": <구현체 이름>}
//!   POST /eta    → {"minutes": <float>, "model": <구현체 이름>}
//! 잘못된 입력은 FastAPI(pydantic) 처럼 422 와 {"detail": [{type, loc, msg, input, ctx?}]} 로 거부한다.
//! pydantic 의 lax 모드와 같게 숫자 문자열("3.5"), true/false, 소수부가 0인 정수(3.0)는 받아들이고
//! 모르는 필드는 무시한다.

use std::num::IntErrorKind;
use std::sync::Arc;

use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::{Map, Value, json};

/// 입력 상한 (api/app.py 의 MAX_DISTANCE_KM, MAX_PENDING_ORDERS 와 같다).
const MAX_DISTANCE_KM: f64 = 1_000_000.0;
const MAX_PENDING_ORDERS: i64 = 1_000_000;

/// (거리 km, 대기 주문 수) → 예상 시간(분).
pub type Predictor = Arc<dyn Fn(f64, i64) -> Result<f64, String> + Send + Sync>;

#[derive(Clone)]
struct AppState {
    model_name: Arc<str>,
    predict: Predictor,
}

/// 계약대로 동작하는 라우터를 만든다. 다른 메서드는 405, 없는 경로는 404 (axum 기본 동작).
pub fn router(model_name: &str, predict: Predictor) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/eta", post(eta))
        .with_state(AppState {
            model_name: model_name.into(),
            predict,
        })
}

async fn health(State(s): State<AppState>) -> Json<Value> {
    Json(json!({"status": "ok", "model": &*s.model_name}))
}

async fn eta(State(s): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let (distance, orders) = match parse_eta_request(&headers, &body) {
        Ok(v) => v,
        Err(detail) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"detail": detail})),
            )
                .into_response();
        }
    };
    match (s.predict)(distance, orders) {
        Ok(minutes) => Json(json!({"minutes": minutes, "model": &*s.model_name})).into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"detail": "Internal Server Error"})),
        )
            .into_response(),
    }
}

fn parse_eta_request(headers: &HeaderMap, body: &[u8]) -> Result<(f64, i64), Vec<Value>> {
    // FastAPI 는 Content-Type 이 JSON 이 아니면 본문을 문자열로 보고 객체가 아니라며 거부한다
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !is_json_content_type(content_type) {
        return Err(vec![not_object(Value::String(
            String::from_utf8_lossy(body).into_owned(),
        ))]);
    }
    let parsed: Value = serde_json::from_slice(body).map_err(|_| {
        vec![json!({"type": "json_invalid", "loc": ["body", 0], "msg": "JSON decode error", "input": {}})]
    })?;
    let Value::Object(fields) = &parsed else {
        return Err(vec![not_object(parsed.clone())]);
    };

    let distance = parse_non_neg_float(fields, "distance_km", &parsed);
    let orders = parse_non_neg_int(fields, "pending_orders", &parsed);
    match (distance, orders) {
        (Ok(d), Ok(n)) => Ok((d, n)),
        (d, n) => Err([d.err(), n.err()].into_iter().flatten().collect()),
    }
}

fn parse_non_neg_float(
    fields: &Map<String, Value>,
    name: &str,
    body: &Value,
) -> Result<f64, Value> {
    let Some(input) = fields.get(name) else {
        return Err(missing(name, body));
    };
    let err = |ty: &str, msg: &str| json!({"type": ty, "loc": ["body", name], "msg": msg, "input": input});
    let v = match input {
        // 원래 숫자 문자열을 직접 파싱한다. 1e309 는 inf 가 되어 아래 finite_number 검사에 걸린다
        Value::Number(n) => n
            .to_string()
            .parse::<f64>()
            .map_err(|_| err("float_parsing", "Input should be a valid number"))?,
        Value::Bool(b) => f64::from(u8::from(*b)),
        Value::String(s) => s.trim().parse::<f64>().map_err(|_| {
            err(
                "float_parsing",
                "Input should be a valid number, unable to parse string as a number",
            )
        })?,
        _ => return Err(err("float_type", "Input should be a valid number")),
    };
    if !v.is_finite() {
        // "inf", "nan" 문자열, 1e309 등. input 은 pydantic 처럼 "inf"/"-inf"/"nan"
        let mut e = err("finite_number", "Input should be a finite number");
        e["input"] = json!(match v {
            v if v.is_nan() => "nan",
            v if v > 0.0 => "inf",
            _ => "-inf",
        });
        return Err(e);
    }
    if v < 0.0 {
        let mut e = err(
            "greater_than_equal",
            "Input should be greater than or equal to 0",
        );
        e["ctx"] = json!({"ge": 0.0});
        return Err(e);
    }
    if v > MAX_DISTANCE_KM {
        let mut e = err(
            "less_than_equal",
            "Input should be less than or equal to 1000000",
        );
        e["ctx"] = json!({"le": MAX_DISTANCE_KM});
        return Err(e);
    }
    Ok(v)
}

fn parse_non_neg_int(fields: &Map<String, Value>, name: &str, body: &Value) -> Result<i64, Value> {
    let Some(input) = fields.get(name) else {
        return Err(missing(name, body));
    };
    let err = |ty: &str, msg: &str| json!({"type": ty, "loc": ["body", name], "msg": msg, "input": input});
    let v = match input {
        Value::Number(n) => match n.as_i64() {
            Some(i) => i,
            None => match n.as_f64() {
                // 소수부가 0 이면 정수. 범위 밖 값은 `as` 가 i64 끝값으로 포화시켜 아래 ge/le 검사에 걸린다
                Some(f) if f.is_finite() && f.fract() == 0.0 => f as i64,
                _ => {
                    return Err(err(
                        "int_from_float",
                        "Input should be a valid integer, got a number with a fractional part",
                    ));
                }
            },
        },
        Value::Bool(b) => i64::from(*b),
        // i64 를 넘는 정수 문자열은 끝값으로 포화시켜 아래 ge/le 검사에 걸리게 한다 (pydantic 과 같음)
        Value::String(s) => match s.trim().parse::<i64>() {
            Ok(i) => i,
            Err(e) if *e.kind() == IntErrorKind::PosOverflow => i64::MAX,
            Err(e) if *e.kind() == IntErrorKind::NegOverflow => i64::MIN,
            Err(_) => {
                return Err(err(
                    "int_parsing",
                    "Input should be a valid integer, unable to parse string as an integer",
                ));
            }
        },
        _ => return Err(err("int_type", "Input should be a valid integer")),
    };
    if v < 0 {
        let mut e = err(
            "greater_than_equal",
            "Input should be greater than or equal to 0",
        );
        e["ctx"] = json!({"ge": 0});
        return Err(e);
    }
    if v > MAX_PENDING_ORDERS {
        let mut e = err(
            "less_than_equal",
            "Input should be less than or equal to 1000000",
        );
        e["ctx"] = json!({"le": MAX_PENDING_ORDERS});
        return Err(e);
    }
    Ok(v)
}

fn missing(name: &str, body: &Value) -> Value {
    json!({"type": "missing", "loc": ["body", name], "msg": "Field required", "input": body})
}

fn not_object(input: Value) -> Value {
    json!({
        "type": "model_attributes_type",
        "loc": ["body"],
        "msg": "Input should be a valid dictionary or object to extract fields from",
        "input": input,
    })
}

fn is_json_content_type(ct: &str) -> bool {
    let mt = ct
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    mt == "application/json" || (mt.starts_with("application/") && mt.ends_with("+json"))
}

// HTTP 계약 테스트 (api/tests/test_app.py 와 같은 경우를 다룬다).
// 예측은 기준선 규칙(10 + 3*거리 + 2*주문)을 쓰는 가짜로 바꿔 ONNX Runtime 없이 돌린다.
#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn app() -> Router {
        router("test", Arc::new(|d, n| Ok(10.0 + 3.0 * d + 2.0 * n as f64)))
    }

    async fn call(
        method: &str,
        uri: &str,
        body: &str,
        content_type: Option<&str>,
    ) -> (StatusCode, Value) {
        let mut req = Request::builder().method(method).uri(uri);
        if let Some(ct) = content_type {
            req = req.header(header::CONTENT_TYPE, ct);
        }
        let res = app()
            .oneshot(req.body(Body::from(body.to_owned())).unwrap())
            .await
            .unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }

    async fn post_json(body: &str) -> (StatusCode, Value) {
        call("POST", "/eta", body, Some("application/json")).await
    }

    /// /health 는 200 과 status=ok, 구현체 이름을 돌려준다.
    #[tokio::test]
    async fn health_returns_ok_and_model() {
        let (status, body) = call("GET", "/health", "", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!({"status": "ok", "model": "test"}));
    }

    /// 정상·경계 입력에 200 과 예상값, minutes/model 두 필드만 돌려준다.
    #[tokio::test]
    async fn eta_known_values() {
        let cases = [
            (r#"{"distance_km": 2.0, "pending_orders": 1}"#, 18.0),
            (r#"{"distance_km": 1.5, "pending_orders": 0}"#, 14.5),
            (r#"{"distance_km": 3, "pending_orders": 0}"#, 19.0), // 정수 거리
            (r#"{"distance_km": 0, "pending_orders": 0}"#, 10.0), // 0
            (
                r#"{"distance_km": 1e6, "pending_orders": 100000}"#,
                3_200_010.0,
            ), // 큰 값
            (r#"{"distance_km": "3.5", "pending_orders": 0}"#, 20.5), // 숫자 문자열 (pydantic lax)
            (r#"{"distance_km": 1, "pending_orders": 3.0}"#, 19.0), // 소수부 0 인 정수
            (r#"{"distance_km": 1, "pending_orders": "3"}"#, 19.0),
            (
                r#"{"distance_km": 1, "pending_orders": 0, "extra": 1}"#,
                13.0,
            ), // 모르는 필드는 무시
            (
                r#"{"distance_km": 1000000, "pending_orders": 1000000}"#,
                5_000_010.0,
            ), // 상한값 자체는 허용
        ];
        for (body, want) in cases {
            let (status, got) = post_json(body).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_eq!(got.as_object().unwrap().len(), 2, "{body}: {got}");
            assert_eq!(got["model"], "test", "{body}");
            assert!(
                (got["minutes"].as_f64().unwrap() - want).abs() < 1e-9,
                "{body}: {got}, want {want}"
            );
        }
    }

    /// 음수·숫자 아님·null·누락·JSON 아님·객체 아님은 422 이고 detail 이 문제 필드만 가리킨다.
    #[tokio::test]
    async fn eta_invalid_returns_422_pointing_to_bad_field() {
        // (본문, Content-Type, detail[].loc 의 마지막 요소들 — "" 는 loc 이 ["body"] 뿐인 경우)
        let json_ct = Some("application/json");
        let cases: [(&str, Option<&str>, &[&str]); 20] = [
            (
                r#"{"distance_km": -1.0, "pending_orders": 0}"#,
                json_ct,
                &["distance_km"],
            ),
            (
                r#"{"distance_km": 1.0, "pending_orders": -1}"#,
                json_ct,
                &["pending_orders"],
            ),
            (
                r#"{"distance_km": -0.001, "pending_orders": 0}"#,
                json_ct,
                &["distance_km"],
            ),
            (
                r#"{"distance_km": "far", "pending_orders": 0}"#,
                json_ct,
                &["distance_km"],
            ),
            (
                r#"{"distance_km": 1.0, "pending_orders": "many"}"#,
                json_ct,
                &["pending_orders"],
            ),
            (
                r#"{"distance_km": 1.0, "pending_orders": 1.5}"#,
                json_ct,
                &["pending_orders"],
            ),
            (
                r#"{"distance_km": null, "pending_orders": 0}"#,
                json_ct,
                &["distance_km"],
            ),
            (r#"{"pending_orders": 0}"#, json_ct, &["distance_km"]),
            (r#"{"distance_km": 1.0}"#, json_ct, &["pending_orders"]),
            (r#"{}"#, json_ct, &["distance_km", "pending_orders"]),
            (
                r#"{"distance_km": -1, "pending_orders": "x"}"#,
                json_ct,
                &["distance_km", "pending_orders"],
            ),
            (
                r#"{"distance_km": "inf", "pending_orders": 0}"#,
                json_ct,
                &["distance_km"],
            ), // 무한대 문자열
            (
                r#"{"distance_km": "Infinity", "pending_orders": 0}"#,
                json_ct,
                &["distance_km"],
            ), // 무한대 문자열
            (
                r#"{"distance_km": "nan", "pending_orders": 0}"#,
                json_ct,
                &["distance_km"],
            ), // NaN 문자열
            (
                r#"{"distance_km": 1000000.5, "pending_orders": 0}"#,
                json_ct,
                &["distance_km"],
            ), // 상한 초과
            (
                r#"{"distance_km": 1, "pending_orders": 1000001}"#,
                json_ct,
                &["pending_orders"],
            ), // 상한 초과
            (
                r#"{"distance_km": 1, "pending_orders": 1e20}"#,
                json_ct,
                &["pending_orders"],
            ), // i64 범위 초과
            ("not json", json_ct, &["0"]),
            ("[1, 2]", json_ct, &[""]),
            (r#"{"distance_km": 1, "pending_orders": 0}"#, None, &[""]), // Content-Type 없음
        ];
        for (body, ct, want) in cases {
            let (status, got) = call("POST", "/eta", body, ct).await;
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
            let locs: Vec<String> = got["detail"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| {
                    let loc = d["loc"].as_array().unwrap();
                    if loc.len() == 1 {
                        String::new()
                    } else {
                        loc.last().unwrap().to_string().trim_matches('"').to_owned()
                    }
                })
                .collect();
            assert_eq!(locs, want, "{body}: {got}");
        }
    }

    /// 범위 밖 값은 api/app.py(pydantic) 와 같은 type·input 으로 거부한다.
    #[tokio::test]
    async fn eta_422_type_matches_pydantic() {
        let cases = [
            (
                r#"{"distance_km": 1e309, "pending_orders": 0}"#,
                "finite_number",
                json!("inf"),
            ),
            (
                r#"{"distance_km": -1e309, "pending_orders": 0}"#,
                "finite_number",
                json!("-inf"),
            ),
            (
                r#"{"distance_km": "nan", "pending_orders": 0}"#,
                "finite_number",
                json!("nan"),
            ),
            (
                r#"{"distance_km": 1, "pending_orders": "99999999999999999999"}"#,
                "less_than_equal",
                json!("99999999999999999999"),
            ),
            (
                r#"{"distance_km": 1, "pending_orders": "-99999999999999999999"}"#,
                "greater_than_equal",
                json!("-99999999999999999999"),
            ),
        ];
        for (body, want_type, want_input) in cases {
            let (status, got) = post_json(body).await;
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
            let detail = got["detail"].as_array().unwrap();
            assert_eq!(detail.len(), 1, "{body}: {got}");
            assert_eq!(detail[0]["type"], want_type, "{body}: {got}");
            assert_eq!(detail[0]["input"], want_input, "{body}: {got}");
        }
    }

    /// /eta 에 GET, /health 에 POST 는 405 다.
    #[tokio::test]
    async fn method_not_allowed() {
        assert_eq!(
            call("GET", "/eta", "", None).await.0,
            StatusCode::METHOD_NOT_ALLOWED
        );
        assert_eq!(
            call("POST", "/health", "", None).await.0,
            StatusCode::METHOD_NOT_ALLOWED
        );
    }
}
