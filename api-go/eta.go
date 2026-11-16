package main

// api/app.py 와 같은 HTTP 계약을 구현한다.
//   GET  /health → {"status": "ok", "model": <구현체 이름>}
//   POST /eta    → {"minutes": <float>, "model": <구현체 이름>}
// 잘못된 입력은 FastAPI(pydantic) 처럼 422 와 {"detail": [{type, loc, msg, input, ctx?}]} 로 거부한다.
// pydantic 의 lax 모드와 같게 숫자 문자열("3.5"), true/false, 소수부가 0인 정수(3.0)는 받아들이고
// 모르는 필드는 무시한다.

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"mime"
	"net/http"
	"strconv"
	"strings"
)

// Predictor 는 (거리 km, 대기 주문 수) → 예상 시간(분) 을 계산한다.
type Predictor func(distanceKm float64, pendingOrders int64) (float64, error)

type validationError struct {
	Type  string          `json:"type"`
	Loc   []any           `json:"loc"`
	Msg   string          `json:"msg"`
	Input json.RawMessage `json:"input"`
	Ctx   map[string]any  `json:"ctx,omitempty"`
}

// NewHandler 는 계약대로 동작하는 HTTP 핸들러를 만든다.
func NewHandler(modelName string, predict Predictor) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "model": modelName})
	})
	mux.HandleFunc("POST /eta", func(w http.ResponseWriter, r *http.Request) {
		distance, orders, errs := parseEtaRequest(r)
		if len(errs) > 0 {
			writeJSON(w, http.StatusUnprocessableEntity, map[string]any{"detail": errs})
			return
		}
		minutes, err := predict(distance, orders)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"detail": "Internal Server Error"})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"minutes": minutes, "model": modelName})
	})
	return mux
}

// 입력 상한 (api/app.py 의 MAX_DISTANCE_KM, MAX_PENDING_ORDERS 와 같다)
const (
	maxDistanceKm    = 1_000_000
	maxPendingOrders = 1_000_000
)

func parseEtaRequest(r *http.Request) (float64, int64, []validationError) {
	body, err := io.ReadAll(http.MaxBytesReader(nil, r.Body, 1<<20))
	if err != nil {
		return 0, 0, []validationError{jsonInvalid()}
	}

	// FastAPI 는 Content-Type 이 JSON 이 아니면 본문을 문자열로 보고 객체가 아니라며 거부한다
	if !isJSONContentType(r.Header.Get("Content-Type")) {
		return 0, 0, []validationError{notObject(mustMarshal(string(body)))}
	}
	// 본문은 한 번만 파싱한다. UseNumber 로 숫자 원문을 보존해 정수/실수를 구분한다.
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var parsed any
	if err := dec.Decode(&parsed); err != nil || dec.Decode(new(any)) != io.EOF { // 뒤에 다른 값이 붙어도 거부
		return 0, 0, []validationError{jsonInvalid()}
	}
	fields, ok := parsed.(map[string]any)
	if !ok {
		return 0, 0, []validationError{notObject(json.RawMessage(bytes.TrimSpace(body)))}
	}

	var errs []validationError
	distance, e := parseDistance(fields, "distance_km", body)
	if e != nil {
		errs = append(errs, *e)
	}
	orders, e := parsePendingOrders(fields, "pending_orders", body)
	if e != nil {
		errs = append(errs, *e)
	}
	return distance, orders, errs
}

// parseDistance 는 0 이상 maxDistanceKm 이하의 유한한 실수를 받는다.
func parseDistance(fields map[string]any, name string, body []byte) (float64, *validationError) {
	value, ok := fields[name]
	if !ok {
		return 0, missing(name, body)
	}
	in := mustMarshal(value)
	loc := []any{"body", name}
	var v float64
	switch x := value.(type) {
	case json.Number:
		f, err := x.Float64()
		if err != nil { // 1e309 처럼 float64 범위를 넘으면 무한대 → pydantic 과 같이 finite_number
			return 0, &validationError{Type: "finite_number", Loc: loc, Msg: "Input should be a finite number", Input: in}
		}
		v = f
	case bool:
		v = boolToFloat(x)
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(x), 64)
		if err != nil && !math.IsInf(f, 0) {
			return 0, &validationError{Type: "float_parsing", Loc: loc, Msg: "Input should be a valid number, unable to parse string as a number", Input: in}
		}
		v = f
	default:
		return 0, &validationError{Type: "float_type", Loc: loc, Msg: "Input should be a valid number", Input: in}
	}
	switch {
	case math.IsInf(v, 0) || math.IsNaN(v): // "inf", "nan" 문자열 등
		return 0, &validationError{Type: "finite_number", Loc: loc, Msg: "Input should be a finite number", Input: in}
	case v < 0:
		return 0, &validationError{Type: "greater_than_equal", Loc: loc, Msg: "Input should be greater than or equal to 0", Input: in, Ctx: map[string]any{"ge": 0.0}}
	case v > maxDistanceKm:
		return 0, &validationError{Type: "less_than_equal", Loc: loc, Msg: "Input should be less than or equal to 1000000", Input: in, Ctx: map[string]any{"le": float64(maxDistanceKm)}}
	}
	return v, nil
}

// parsePendingOrders 는 0 이상 maxPendingOrders 이하의 정수를 받는다.
func parsePendingOrders(fields map[string]any, name string, body []byte) (int64, *validationError) {
	value, ok := fields[name]
	if !ok {
		return 0, missing(name, body)
	}
	in := mustMarshal(value)
	loc := []any{"body", name}
	var v int64
	switch x := value.(type) {
	case json.Number:
		if i, err := x.Int64(); err == nil {
			v = i
			break
		}
		f, _ := x.Float64() // int64 를 넘는 정수는 큰 실수(범위를 넘으면 ±Inf)가 된다
		if f != math.Trunc(f) {
			return 0, &validationError{Type: "int_from_float", Loc: loc, Msg: "Input should be a valid integer, got a number with a fractional part", Input: in}
		}
		// 범위 밖 값은 아래 ge/le 검사에 걸리도록 -1 또는 상한+1 로 줄인다 (int64 변환 overflow 방지)
		v = int64(math.Max(-1, math.Min(f, maxPendingOrders+1)))
	case bool:
		v = int64(boolToFloat(x))
	case string:
		i, err := strconv.ParseInt(strings.TrimSpace(x), 10, 64)
		if err != nil {
			return 0, &validationError{Type: "int_parsing", Loc: loc, Msg: "Input should be a valid integer, unable to parse string as an integer", Input: in}
		}
		v = i
	default:
		return 0, &validationError{Type: "int_type", Loc: loc, Msg: "Input should be a valid integer", Input: in}
	}
	switch {
	case v < 0:
		return 0, &validationError{Type: "greater_than_equal", Loc: loc, Msg: "Input should be greater than or equal to 0", Input: in, Ctx: map[string]any{"ge": 0}}
	case v > maxPendingOrders:
		return 0, &validationError{Type: "less_than_equal", Loc: loc, Msg: "Input should be less than or equal to 1000000", Input: in, Ctx: map[string]any{"le": maxPendingOrders}}
	}
	return v, nil
}

func jsonInvalid() validationError {
	return validationError{Type: "json_invalid", Loc: []any{"body", 0}, Msg: "JSON decode error", Input: json.RawMessage(`{}`)}
}

func missing(name string, body []byte) *validationError {
	return &validationError{Type: "missing", Loc: []any{"body", name}, Msg: "Field required", Input: json.RawMessage(bytes.TrimSpace(body))}
}

func notObject(input json.RawMessage) validationError {
	return validationError{Type: "model_attributes_type", Loc: []any{"body"}, Msg: "Input should be a valid dictionary or object to extract fields from", Input: input}
}

func isJSONContentType(ct string) bool {
	mt, _, err := mime.ParseMediaType(ct)
	if err != nil {
		return false
	}
	return mt == "application/json" || (strings.HasPrefix(mt, "application/") && strings.HasSuffix(mt, "+json"))
}

func boolToFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

func mustMarshal(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
