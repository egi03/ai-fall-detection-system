# Sample videos

Place short test clips here to try the demo without a webcam or full dataset.

Recommended:
- `fall_sample.mp4` — a 10–15 second clip containing a fall event
- `walking_sample.mp4` — a 10–15 second clip of normal walking (ADL)

Record your own clips or extract sequences from the URFD/Le2i datasets.
Keep clips under 25 MB (use `ffmpeg -crf 28` to compress).

Run the demo:
```bash
python scripts/demo.py --source samples/fall_sample.mp4 --model models/run5_stride2_aug
```
