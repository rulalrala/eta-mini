// Go 로 구현한 배달 예상 시간(ETA) API. api/app.py 와 같은 계약을 따른다.
// 예측은 학습 파이프라인이 만든 ONNX 모델(model_sklearn.onnx)을 ONNX Runtime 으로 실행한다.
// 정규화는 모델 안에 있으므로 원래 단위의 값을 그대로 넣는다.
//
// 환경 변수
//
//	PORT          듣는 포트 (기본 8080)
//	MODEL_PATH    ONNX 모델 경로 (기본 artifacts/model_sklearn.onnx)
//	ORT_LIB_PATH  ONNX Runtime 공유 라이브러리 경로 (기본 libonnxruntime.so)
//	MODEL_NAME    /health, /eta 응답의 model 값 (기본 go-onnx)
package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// newOnnxPredictor 는 모델을 한 번 읽어 두고, 요청마다 입력 "X" [1, 2] float32 → 출력 "variable" 을 실행한다.
// DynamicAdvancedSession 은 호출마다 텐서를 따로 만들므로 여러 요청이 동시에 Run 해도 안전하다.
func newOnnxPredictor(libPath, modelPath string) (Predictor, func(), error) {
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, nil, fmt.Errorf("ONNX Runtime 초기화 실패 (%s): %w", libPath, err)
	}
	sess, err := ort.NewDynamicAdvancedSession(modelPath, []string{"X"}, []string{"variable"}, nil)
	if err != nil {
		_ = ort.DestroyEnvironment()
		return nil, nil, fmt.Errorf("모델 로드 실패 (%s): %w", modelPath, err)
	}
	cleanup := func() {
		_ = sess.Destroy()
		_ = ort.DestroyEnvironment()
	}

	predict := func(distanceKm float64, pendingOrders int64) (float64, error) {
		x, err := ort.NewTensor(ort.NewShape(1, 2), []float32{float32(distanceKm), float32(pendingOrders)})
		if err != nil {
			return 0, err
		}
		defer x.Destroy()
		outputs := []ort.Value{nil} // nil 이면 ONNX Runtime 이 출력 텐서를 만들어 준다
		if err := sess.Run([]ort.Value{x}, outputs); err != nil {
			return 0, err
		}
		defer outputs[0].Destroy()
		y, ok := outputs[0].(*ort.Tensor[float32])
		if !ok || len(y.GetData()) == 0 {
			return 0, fmt.Errorf("예상과 다른 출력: %T", outputs[0])
		}
		return float64(y.GetData()[0]), nil
	}
	return predict, cleanup, nil
}

func main() {
	predict, cleanup, err := newOnnxPredictor(
		getenv("ORT_LIB_PATH", "libonnxruntime.so"),
		getenv("MODEL_PATH", "artifacts/model_sklearn.onnx"),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer cleanup()

	addr := ":" + getenv("PORT", "8080")
	log.Printf("listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, NewHandler(getenv("MODEL_NAME", "go-onnx"), predict)))
}
