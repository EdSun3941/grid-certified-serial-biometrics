# Data

The score files are not redistributed. The scripts look for the processed files in the folder named by the environment
variable `BSSR1_DIR`, else in `data/`, else in `data/processed/`.

## Subsets D1–D3: NIST BSSR1

Download the NIST Biometric Scores Set Release 1 (BSSR1) from
https://www.nist.gov/itl/iad/image-group/nist-biometric-scores-set-bssr1 and convert it:

```
cd code
python prepare_bssr1.py <bssr1_root> ../data/processed chunk <subset> <matcher> <row_start> <row_stop>   # repeat per chunk
python prepare_bssr1.py <bssr1_root> ../data/processed combine <subset>
```

| Subset | File | Content | SHA-256 |
|---|---|---|---|
| D1 | `bssr1_fing_x_face.npz` | 517 subjects; face matchers C and G, left and right index fingers | `25d2b08f196301fd9d9584012838a5b4da9d228e2f74631edf4ecb1c4ed50fe9` |
| D2 | `bssr1_fing_x_fing.npz` | 6000 subjects; left and right index fingers | `27936c42f43f9b012922d2f3e8f846b741b902844aedaa831c78a02824dea928` |
| D3 | `bssr1_face_x_face.npz` | 6000 probes x 3000 enrollees; face matchers C and G | `42bd6ca98015fd9e831e26469317be82b16e00156be00e7c6e56695b43919817` |

Scores are similarities (accept if s >= t). BSSR1 failure scores (-1) are kept as the lowest score.

## Subset D4: LFW faces with BSSR1 fingers (chimeric)

1. Download `lfw.tgz` (Labeled Faces in the Wild, University of Massachusetts Amherst; SHA-256
   `055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0`, as published with scikit-learn) and extract it
   to `data/lfw_src/lfw`.
2. Download `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx` from the OpenCV model zoo
   (https://github.com/opencv/opencv_zoo) into `data/lfw_src`.
3. Build the subset (needs D2 above and `opencv-python` 4.13):

```
cd code
python build_lfw_chimeric.py ../data/lfw_src
```

This writes `lfw_x_fing.npz` (1680 subjects) next to the D2 file, and a build log `lfw_x_fing_build.json`. The build log of
the file used for the article is included here; it records the SHA-256 values of the two models and of the output
(`9bb835b36905d0467cede2629e0ff134396be27a085338f0a291ae77ea1c6cd1`). Only similarity scores are computed and kept;
no images are stored.
