package main

// HTTP 계약 테스트 (api/tests/test_app.py 와 같은 경우를 다룬다).
// 예측은 기준선 규칙(10 + 3*거리 + 2*주문)을 쓰는 가짜로 바꿔 ONNX Runtime 없이 돌린다.

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func ruleHandler() http.Handler {
	return NewHandler("test", func(d float64, n int64) (float64, error) { return 10 + 3*d + 2*float64(n), nil })
}

func post(t *testing.T, body, contentType string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/eta", strings.NewReader(body))
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	rec := httptest.NewRecorder()
	ruleHandler().ServeHTTP(rec, req)
	return rec
}

// TestHealth: /health 는 200 과 status=ok, 구현체 이름을 돌려준다.
func TestHealth(t *testing.T) {
	rec := httptest.NewRecorder()
	ruleHandler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))
	if rec.Code != 200 || strings.TrimSpace(rec.Body.String()) != `{"model":"test","status":"ok"}` {
		t.Fatalf("got %d %s", rec.Code, rec.Body)
	}
}

// TestEtaKnownValues: 정상·경계 입력에 200 과 예상값, minutes/model 두 필드만 돌려준다.
func TestEtaKnownValues(t *testing.T) {
	cases := []struct {
		body string
		want float64
	}{
		{`{"distance_km": 2.0, "pending_orders": 1}`, 18},
		{`{"distance_km": 1.5, "pending_orders": 0}`, 14.5},
		{`{"distance_km": 3, "pending_orders": 0}`, 19},             // 정수 거리
		{`{"distance_km": 0, "pending_orders": 0}`, 10},             // 0
		{`{"distance_km": 1e6, "pending_orders": 100000}`, 3200010}, // 큰 값
		{`{"distance_km": "3.5", "pending_orders": 0}`, 20.5},       // 숫자 문자열 (pydantic lax)
		{`{"distance_km": 1, "pending_orders": 3.0}`, 19},           // 소수부 0 인 정수
		{`{"distance_km": 1, "pending_orders": "3"}`, 19},
		{`{"distance_km": 1, "pending_orders": 0, "extra": 1}`, 13}, // 모르는 필드는 무시
		{`{"distance_km": 1000000, "pending_orders": 1000000}`, 5000010}, // 상한값 자체는 허용
	}
	for _, c := range cases {
		rec := post(t, c.body, "application/json")
		var got map[string]any
		_ = json.Unmarshal(rec.Body.Bytes(), &got)
		if rec.Code != 200 || len(got) != 2 || got["model"] != "test" || math.Abs(got["minutes"].(float64)-c.want) > 1e-9 {
			t.Errorf("%s: got %d %s, want minutes %v", c.body, rec.Code, rec.Body, c.want)
		}
	}
}

// TestEtaInvalidReturns422: 음수·숫자 아님·null·누락·JSON 아님·객체 아님은 422 이고 detail 이 문제 필드만 가리킨다.
func TestEtaInvalidReturns422(t *testing.T) {
	cases := []struct {
		body, contentType string
		wantLoc           []string // detail[].loc 의 마지막 요소들 ("" 는 loc 이 ["body"] 뿐인 경우)
	}{
		{`{"distance_km": -1.0, "pending_orders": 0}`, "application/json", []string{"distance_km"}},
		{`{"distance_km": 1.0, "pending_orders": -1}`, "application/json", []string{"pending_orders"}},
		{`{"distance_km": -0.001, "pending_orders": 0}`, "application/json", []string{"distance_km"}},
		{`{"distance_km": "far", "pending_orders": 0}`, "application/json", []string{"distance_km"}},
		{`{"distance_km": 1.0, "pending_orders": "many"}`, "application/json", []string{"pending_orders"}},
		{`{"distance_km": 1.0, "pending_orders": 1.5}`, "application/json", []string{"pending_orders"}},
		{`{"distance_km": null, "pending_orders": 0}`, "application/json", []string{"distance_km"}},
		{`{"pending_orders": 0}`, "application/json", []string{"distance_km"}},
		{`{"distance_km": 1.0}`, "application/json", []string{"pending_orders"}},
		{`{}`, "application/json", []string{"distance_km", "pending_orders"}},
		{`{"distance_km": -1, "pending_orders": "x"}`, "application/json", []string{"distance_km", "pending_orders"}},
		{`{"distance_km": "inf", "pending_orders": 0}`, "application/json", []string{"distance_km"}},      // 무한대 문자열
		{`{"distance_km": "Infinity", "pending_orders": 0}`, "application/json", []string{"distance_km"}}, // 무한대 문자열
		{`{"distance_km": "nan", "pending_orders": 0}`, "application/json", []string{"distance_km"}},      // NaN 문자열
		{`{"distance_km": 1e309, "pending_orders": 0}`, "application/json", []string{"distance_km"}},      // float64 범위 초과
		{`{"distance_km": 1000000.5, "pending_orders": 0}`, "application/json", []string{"distance_km"}},  // 상한 초과
		{`{"distance_km": 1, "pending_orders": 1000001}`, "application/json", []string{"pending_orders"}},  // 상한 초과
		{`{"distance_km": 1, "pending_orders": ` + strings.Repeat("9", 400) + `}`, "application/json", []string{"pending_orders"}}, // int64 범위 초과
		{`{"distance_km": 1, "pending_orders": 0} {}`, "application/json", []string{"0"}}, // JSON 값 뒤에 다른 값
		{`not json`, "application/json", []string{"0"}},
		{`[1, 2]`, "application/json", []string{""}},
		{`{"distance_km": 1, "pending_orders": 0}`, "", []string{""}}, // Content-Type 없음
	}
	for _, c := range cases {
		rec := post(t, c.body, c.contentType)
		var got struct {
			Detail []struct {
				Loc []any `json:"loc"`
			} `json:"detail"`
		}
		_ = json.Unmarshal(rec.Body.Bytes(), &got)
		var locs []string
		for _, d := range got.Detail {
			if len(d.Loc) == 1 {
				locs = append(locs, "")
			} else {
				locs = append(locs, strings.Trim(mustString(d.Loc[len(d.Loc)-1]), `"`))
			}
		}
		if rec.Code != 422 || strings.Join(locs, ",") != strings.Join(c.wantLoc, ",") {
			t.Errorf("%q: got %d %s, want 422 with loc %v", c.body, rec.Code, rec.Body, c.wantLoc)
		}
	}
}

// TestMethodNotAllowed: /eta 에 GET, /health 에 POST 는 405 다.
func TestMethodNotAllowed(t *testing.T) {
	for _, r := range []*http.Request{
		httptest.NewRequest(http.MethodGet, "/eta", nil),
		httptest.NewRequest(http.MethodPost, "/health", nil),
	} {
		rec := httptest.NewRecorder()
		ruleHandler().ServeHTTP(rec, r)
		if rec.Code != 405 {
			t.Errorf("%s %s: got %d, want 405", r.Method, r.URL.Path, rec.Code)
		}
	}
}

func mustString(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}
