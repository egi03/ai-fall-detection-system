## Makefile for the fall-detection demo / training shortcuts.
##
## Run `make help` to list every target. All demo targets accept SOURCE,
## defaulting to video.avi in the repo root. Override per invocation:
##
##     make demo                            # best model on video.avi
##     make demo SOURCE=samples/foo.mp4     # different clip
##     make demo SOURCE=0                   # webcam (device index 0)
##
## Requires `make` on PATH. On Windows the easiest way is to install
## chocolatey then `choco install make`, or run from a Git-Bash shell.

PY        := venv/Scripts/python.exe
DEMO      := $(PY) scripts/demo.py
SOURCE    ?= video.avi

# Model paths
LSTM_R3      := models/urfd
LSTM_R5      := models/run5_stride2_aug
STGCN_MOTION := models/stgcn_motion
STGCN_V1     := models/stgcn_loso

# Common alarm tuning for live demos
DEMO_FLAGS := --demo-mode

.DEFAULT_GOAL := help

.PHONY: help demo demo-best demo-lstm demo-lstm-single demo-stgcn demo-stgcn-baseline \
        demo-explain demo-vlm demo-skeleton demo-webcam \
        test sweep sweep-stgcn-motion sweep-stgcn-baseline \
        events events-clear \
        train-stgcn train-stgcn-baseline install-vlm clean-runs

## help: List every target with its one-line description
help:
	@echo "Fall-detection demo Makefile"
	@echo ""
	@echo "Run a demo (override video with SOURCE=...):"
	@echo "  make demo                  - BEST: LSTM ensemble + explain + vlm + demo-mode"
	@echo "  make demo-best             - alias for 'demo'"
	@echo "  make demo-lstm             - LSTM ensemble (R3 + R5), no explain/vlm"
	@echo "  make demo-lstm-single      - LSTM single (Run 5, AUC 0.888)"
	@echo "  make demo-stgcn            - ST-GCN motion, AUC 0.862 (graph baseline)"
	@echo "  make demo-stgcn-baseline   - ST-GCN v1, AUC 0.847 (original 3-channel)"
	@echo "  make demo-explain          - LSTM ensemble + Integrated Gradients only"
	@echo "  make demo-vlm              - LSTM ensemble + Gemini narration only"
	@echo "  make demo-skeleton         - no model, skeleton-only overlay"
	@echo "  make demo-webcam           - LSTM ensemble + everything from webcam 0"
	@echo ""
	@echo "Tests + analysis:"
	@echo "  make test                  - pytest -x -q"
	@echo "  make sweep                 - threshold sweep on both ST-GCN runs"
	@echo "  make events                - dump last 10 fall events from logs/events.db"
	@echo "  make events-clear          - delete logs/events.db"
	@echo ""
	@echo "Training (slow):"
	@echo "  make train-stgcn           - LOSO train ST-GCN motion (new canonical)"
	@echo "  make train-stgcn-baseline  - LOSO train ST-GCN v1 (3-channel)"
	@echo ""
	@echo "Setup:"
	@echo "  make install-vlm           - pip install google-genai for --vlm"

## demo: Best showpiece — LSTM ensemble + Integrated Gradients + Gemini narration
demo: demo-best

demo-best:
	$(DEMO) --source $(SOURCE) --model $(LSTM_R3) --model2 $(LSTM_R5) \
	        --arch lstm --explain --vlm $(DEMO_FLAGS)

## demo-lstm: LSTM ensemble only — no explainability, no VLM (fastest baseline)
demo-lstm:
	$(DEMO) --source $(SOURCE) --model $(LSTM_R3) --model2 $(LSTM_R5) \
	        --arch lstm $(DEMO_FLAGS)

## demo-lstm-single: Single best BiLSTM (Run 5, stride=2 + augment)
demo-lstm-single:
	$(DEMO) --source $(SOURCE) --model $(LSTM_R5) --arch lstm $(DEMO_FLAGS)

## demo-stgcn: ST-GCN motion variant (canonical graph baseline)
demo-stgcn:
	$(DEMO) --source $(SOURCE) --model $(STGCN_MOTION) --arch stgcn --vlm $(DEMO_FLAGS)

## demo-stgcn-baseline: Original ST-GCN (3-channel, no motion)
demo-stgcn-baseline:
	$(DEMO) --source $(SOURCE) --model $(STGCN_V1) --arch stgcn $(DEMO_FLAGS)

## demo-explain: LSTM ensemble + Integrated Gradients overlay (no VLM)
demo-explain:
	$(DEMO) --source $(SOURCE) --model $(LSTM_R3) --model2 $(LSTM_R5) \
	        --arch lstm --explain $(DEMO_FLAGS)

## demo-vlm: LSTM ensemble + Gemini fall narration (no IG overlay)
demo-vlm:
	$(DEMO) --source $(SOURCE) --model $(LSTM_R3) --model2 $(LSTM_R5) \
	        --arch lstm --vlm $(DEMO_FLAGS)

## demo-skeleton: No model — just pose overlay for sanity-checking the pipeline
demo-skeleton:
	$(DEMO) --source $(SOURCE) $(DEMO_FLAGS)

## demo-webcam: LSTM ensemble + everything, reading from webcam 0
demo-webcam:
	$(DEMO) --source 0 --model $(LSTM_R3) --model2 $(LSTM_R5) \
	        --arch lstm --explain --vlm $(DEMO_FLAGS)

## test: Run the pytest suite
test:
	$(PY) -m pytest tests/ -x -q

## sweep: Threshold sweep on every saved ST-GCN run
sweep: sweep-stgcn-motion sweep-stgcn-baseline

sweep-stgcn-motion:
	$(PY) scripts/stgcn_threshold_sweep.py --run stgcn_motion

sweep-stgcn-baseline:
	$(PY) scripts/stgcn_threshold_sweep.py --run stgcn_loso

## events: Print the most recent fall events from the SQLite alarm log
events:
	$(PY) -c "from src.alarm.logger import EventLogger; from pathlib import Path; \
	          el = EventLogger(Path('logs/events.db')); \
	          [print(dict(e)) for e in el.get_events(limit=10)]; el.close()"

events-clear:
	rm -f logs/events.db logs/events.db-wal logs/events.db-shm

## train-stgcn: Train the canonical ST-GCN (motion channels, no augment) LOSO
train-stgcn:
	$(PY) scripts/train_stgcn.py --epochs 80 --batch-size 32 --stride 2 \
	      --run-name stgcn_motion --motion --device cpu

## train-stgcn-baseline: Retrain the original 3-channel ST-GCN
train-stgcn-baseline:
	$(PY) scripts/train_stgcn.py --epochs 80 --batch-size 32 --stride 2 \
	      --run-name stgcn_loso --device cpu

## install-vlm: One-off install of the Gemini SDK used by --vlm
install-vlm:
	venv/Scripts/pip.exe install google-genai
