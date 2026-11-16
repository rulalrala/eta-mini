//! Rust 로 구현한 배달 예상 시간(ETA) API. api/app.py 와 같은 계약을 따른다.
//! 예측은 학습 파이프라인이 만든 ONNX 모델(model_sklearn.onnx)을 ONNX Runtime 으로 실행한다.
//! 정규화는 모델 안에 있으므로 원래 단위의 값을 그대로 넣는다.
//!
//! 환경 변수
//!   PORT          듣는 포트 (기본 8080)
//!   MODEL_PATH    ONNX 모델 경로 (기본 artifacts/model_sklearn.onnx)
//!   ORT_LIB_PATH  ONNX Runtime 공유 라이브러리 경로 (기본 libonnxruntime.so)
//!   MODEL_NAME    /health, /eta 응답의 model 값 (기본 rust-onnx)

mod api;

use std::sync::{Arc, Mutex};

use ort::{session::Session, value::Tensor};

fn getenv(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_owned())
}

/// 모델을 한 번 읽어 두고, 요청마다 입력 "X" [1, 2] float32 → 출력 "variable" 을 실행한다.
/// Session::run 이 &mut self 를 받으므로 Mutex 로 감싼다 (모델이 작아 잠금 시간은 무시할 만하다).
fn onnx_predictor(lib_path: &str, model_path: &str) -> Result<api::Predictor, String> {
    let env = ort::init_from(lib_path)
        .map_err(|e| e.to_string())
        .and_then(|b| b.build().map_err(|e| e.to_string()))
        .map_err(|e| format!("ONNX Runtime 초기화 실패 ({lib_path}): {e}"))?;
    let session = Session::builder(&env)
        .and_then(|mut b| b.commit_from_file(model_path))
        .map_err(|e| format!("모델 로드 실패 ({model_path}): {e}"))?;
    let session = Mutex::new(session);

    Ok(Arc::new(move |distance_km: f64, pending_orders: i64| {
        let _ = &env; // 세션이 쓰는 Environment 를 서버가 끝날 때까지 살려 둔다
        let x = Tensor::from_array(([1usize, 2], vec![distance_km as f32, pending_orders as f32]))
            .map_err(|e| e.to_string())?;
        let mut session = session.lock().map_err(|e| e.to_string())?;
        let outputs = session
            .run(ort::inputs!["X" => x])
            .map_err(|e| e.to_string())?;
        let (_, data) = outputs["variable"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        data.first()
            .map(|&y| f64::from(y))
            .ok_or_else(|| "빈 출력".to_owned())
    }))
}

#[tokio::main]
async fn main() {
    let predict = onnx_predictor(
        &getenv("ORT_LIB_PATH", "libonnxruntime.so"),
        &getenv("MODEL_PATH", "artifacts/model_sklearn.onnx"),
    )
    .unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    let addr = format!("0.0.0.0:{}", getenv("PORT", "8080"));
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("포트를 열 수 없음");
    println!("listening on {addr}");
    axum::serve(
        listener,
        api::router(&getenv("MODEL_NAME", "rust-onnx"), predict),
    )
    .await
    .expect("서버 오류");
}
