# Results of the CPU experiments

Made by `docs/experiments/make_figures.py` from the run folders; what each experiment does and how to run it: [HOW_IT_WORKS.md](../HOW_IT_WORKS.md#12-experiments-run-on-the-cpu).

## Sensor-noise sweeps (stand-in mission, world seed 0)

**Position drift** (std of the position drift [m])

| Level | Seeds | People found | False alarms | Pin error [m] | Pin error, people [m] |
|---|---|---|---|---|---|
| 0 | 3 | 3.0 / 3 | 0.0 ± 0.0 | 0.047 ± 0.000 | 0.047 ± 0.000 |
| 0.1 | 3 | 3.0 / 3 | 0.0 ± 0.0 | 0.116 ± 0.061 | 0.098 ± 0.038 |
| 0.25 | 3 | 3.0 / 3 | 0.0 ± 0.0 | 0.262 ± 0.163 | 0.211 ± 0.102 |
| 0.5 | 3 | 3.0 / 3 | 0.3 ± 0.6 | 0.513 ± 0.330 | 0.397 ± 0.195 |
| 1 | 3 | 3.0 / 3 | 3.0 ± 2.0 | 0.781 ± 0.353 | 0.751 ± 0.398 |
| 2 | 3 | 2.3 / 3 | 11.3 ± 5.0 | 1.016 ± 0.318 | 0.920 ± 0.320 |

**Attitude drift** (std of the attitude drift [deg])

| Level | Seeds | People found | False alarms | Pin error [m] | Pin error, people [m] |
|---|---|---|---|---|---|
| 0 | 3 | 3.0 / 3 | 0.0 ± 0.0 | 0.047 ± 0.000 | 0.047 ± 0.000 |
| 0.5 | 3 | 3.0 / 3 | 0.0 ± 0.0 | 0.087 ± 0.026 | 0.074 ± 0.033 |
| 1 | 3 | 3.0 / 3 | 0.0 ± 0.0 | 0.152 ± 0.045 | 0.130 ± 0.058 |
| 2 | 3 | 3.0 / 3 | 0.0 ± 0.0 | 0.291 ± 0.077 | 0.254 ± 0.110 |
| 5 | 3 | 3.0 / 3 | 3.3 ± 1.5 | 0.766 ± 0.110 | 0.690 ± 0.246 |

**Depth noise** (std as a fraction of the depth)

| Level | Seeds | People found | False alarms | Pin error [m] | Pin error, people [m] |
|---|---|---|---|---|---|
| 0 | 3 | 3.0 / 3 | 0.0 ± 0.0 | 0.047 ± 0.000 | 0.047 ± 0.000 |
| 0.01 | 3 | 3.0 / 3 | 0.0 ± 0.0 | 0.046 ± 0.000 | 0.043 ± 0.000 |
| 0.02 | 3 | 3.0 / 3 | 0.0 ± 0.0 | 0.046 ± 0.000 | 0.042 ± 0.000 |
| 0.05 | 3 | 3.0 / 3 | 0.0 ± 0.0 | 0.046 ± 0.000 | 0.042 ± 0.000 |
| 0.1 | 3 | 3.0 / 3 | 0.0 ± 0.0 | 0.047 ± 0.000 | 0.044 ± 0.001 |

**Camera lag** (camera paired with a pose this much later [s])

| Level | Seeds | People found | False alarms | Pin error [m] | Pin error, people [m] |
|---|---|---|---|---|---|
| 0 | 1 | 3.0 / 3 | 0.0 ± 0.0 | 0.047 ± 0.000 | 0.047 ± 0.000 |
| 0.05 | 1 | 3.0 / 3 | 0.0 ± 0.0 | 0.056 ± 0.000 | 0.054 ± 0.000 |
| 0.1 | 1 | 3.0 / 3 | 0.0 ± 0.0 | 0.079 ± 0.000 | 0.075 ± 0.000 |
| 0.2 | 1 | 3.0 / 3 | 0.0 ± 0.0 | 0.139 ± 0.000 | 0.139 ± 0.000 |
| 0.5 | 1 | 3.0 / 3 | 10.0 ± 0.0 | 1.482 ± 0.000 | 1.483 ± 0.000 |

## Stand-in benchmark (world seeds 0-4)

### Stand-in benchmark, without noise

| | |
|---|---|
| Missions | 5 (5 flew the whole plan) |
| People found | 15 / 15 (100.0 %; 95 % interval 79.6-100.0 %, every person counted as one trial) |
| Missions with every person found | 5 / 5 |
| People found per mission | 100.0 ± 0.0 % (lowest 100.0 %) |
| False alarms | 0 in total; 0.00 per mission (most 0); none in 5 / 5 missions |
| Objects found (all classes) | 49 / 50 (98.0 %; 95 % interval 89.5-99.6 %, every object counted as one trial) |
| Pin error | 0.053 ± 0.014 m (0.034 .. 0.069, 5 runs) |
| Pin error, people | 0.047 ± 0.014 m (0.033 .. 0.069, 5 runs) |
| Map height error (RMSE) | - |
| Map height error away from steps | - |
| Scan coverage | 99.9 ± 0.0 % (99.9 .. 99.9, 5 runs) |
| Photo coverage | 100.0 ± 0.0 % (100.0 .. 100.0, 5 runs) |
| Flight time | 172.7 ± 0.0 s (172.7 .. 172.7, 5 runs) |
| Closest approach to obstacles | - |

Each value is the mean ± the standard deviation over the missions (lowest .. highest). The interval of the pooled counts is a Wilson score interval.

## Missions

| Run | Site seed | Terrain seed | End | People | False alarms | Objects | Pin error [m] | Map error [m] | Scanned [%] | Flight [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| exp_flight_s0 | - | - | complete | 3 / 3 | 0 | 10 / 10 | 0.047 | - | 99.9 | 172.7 |
| exp_flight_s1 | - | - | complete | 3 / 3 | 0 | 10 / 10 | 0.064 | - | 99.9 | 172.7 |
| exp_flight_s2 | - | - | complete | 3 / 3 | 0 | 9 / 10 | 0.069 | - | 99.9 | 172.7 |
| exp_flight_s3 | - | - | complete | 3 / 3 | 0 | 10 / 10 | 0.051 | - | 99.9 | 172.7 |
| exp_flight_s4 | - | - | complete | 3 / 3 | 0 | 10 / 10 | 0.034 | - | 99.9 | 172.7 |

### Stand-in benchmark, replayed with pos=0.5, att=1, depth=0.02

| | |
|---|---|
| Missions | 5 (5 flew the whole plan) |
| People found | 15 / 15 (100.0 %; 95 % interval 79.6-100.0 %, every person counted as one trial) |
| Missions with every person found | 5 / 5 |
| People found per mission | 100.0 ± 0.0 % (lowest 100.0 %) |
| False alarms | 0 in total; 0.00 per mission (most 0); none in 5 / 5 missions |
| Objects found (all classes) | 49 / 50 (98.0 %; 95 % interval 89.5-99.6 %, every object counted as one trial) |
| Pin error | 0.820 ± 0.064 m (0.737 .. 0.890, 5 runs) |
| Pin error, people | 0.733 ± 0.299 m (0.495 .. 1.239, 5 runs) |
| Map height error (RMSE) | - |
| Map height error away from steps | - |
| Scan coverage | 99.8 ± 0.0 % (99.8 .. 99.8, 5 runs) |
| Photo coverage | 100.0 ± 0.0 % (100.0 .. 100.0, 5 runs) |
| Flight time | 172.7 ± 0.0 s (172.7 .. 172.7, 5 runs) |
| Closest approach to obstacles | - |

Each value is the mean ± the standard deviation over the missions (lowest .. highest). The interval of the pooled counts is a Wilson score interval.

## Missions

| Run | Site seed | Terrain seed | End | People | False alarms | Objects | Pin error [m] | Map error [m] | Scanned [%] | Flight [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| exp_noisy_s0 | - | - | complete | 3 / 3 | 0 | 10 / 10 | 0.846 | - | 99.8 | 172.7 |
| exp_noisy_s1 | - | - | complete | 3 / 3 | 0 | 10 / 10 | 0.890 | - | 99.8 | 172.7 |
| exp_noisy_s2 | - | - | complete | 3 / 3 | 0 | 9 / 10 | 0.769 | - | 99.8 | 172.7 |
| exp_noisy_s3 | - | - | complete | 3 / 3 | 0 | 10 / 10 | 0.856 | - | 99.8 | 172.7 |
| exp_noisy_s4 | - | - | complete | 3 / 3 | 0 | 10 / 10 | 0.737 | - | 99.8 | 172.7 |

## People on real drone photos (VisDrone2019-DET val)

| Detector | Images given | Images | People labelled | Found | Missed | False alarms | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| fine-tuned on simulator images | whole image (all 548 images) | 548 | 13969 | 227 | 13742 | 3616 | 0.059 | 0.016 |
| fine-tuned on simulator images | whole image (first 50) | 50 | 1451 | 62 | 1389 | 227 | 0.214 | 0.043 |
| fine-tuned on simulator images | tiles, x1 (first 50) | 50 | 1451 | 157 | 1294 | 648 | 0.195 | 0.108 |
| fine-tuned on simulator images | tiles, x2 (first 50) | 50 | 1451 | 161 | 1290 | 1127 | 0.125 | 0.111 |
| stock COCO YOLO26n | whole image (all 548 images) | 548 | 13969 | 936 | 13033 | 139 | 0.871 | 0.067 |
| stock COCO YOLO26n | whole image (first 50) | 50 | 1451 | 272 | 1179 | 24 | 0.919 | 0.188 |
| stock COCO YOLO26n | tiles, x1 (first 50) | 50 | 1451 | 507 | 944 | 76 | 0.870 | 0.349 |
| stock COCO YOLO26n | tiles, x2 (first 50) | 50 | 1451 | 748 | 703 | 157 | 0.827 | 0.515 |

Recall by label size, fine-tuned on simulator images, whole image (all 548):

| Longer side [px] | People labelled | Found | Recall |
|---|---|---|---|
| <16 | 2925 | 0 | 0.000 |
| 16-32 | 6650 | 19 | 0.003 |
| 32-64 | 4043 | 125 | 0.031 |
| 64-128 | 310 | 63 | 0.203 |
| >=128 | 41 | 20 | 0.488 |

Recall by label size, stock COCO YOLO26n, whole image (all 548):

| Longer side [px] | People labelled | Found | Recall |
|---|---|---|---|
| <16 | 2925 | 1 | 0.000 |
| 16-32 | 6650 | 123 | 0.018 |
| 32-64 | 4043 | 587 | 0.145 |
| 64-128 | 310 | 186 | 0.600 |
| >=128 | 41 | 39 | 0.951 |

## Surface-map obstacles (stand-in report of the demo site)

| | Cells |
|---|---|
| Marked by the surface map (more than 0.2 m above the elevation map) | 1967 |
| ... of them open ground for the terrain limits and the pins | 248 |
| No-go cells only with the surface map, after the robot-radius inflation | 326 |
| No-go cells without / with the surface map | 22243 / 22569 |
| Grid | 209 x 209 cells of 0.25 m |

## Rescue tour (ground truth of the demo site)

| Leg | Person | Stop to person [m] | Length [m] | Cost | Walking time [s] |
|---|---|---|---|---|---|
| 1 | {'id': 'person_10', 'class': 'person', 'x': -21.895, 'y': -23.605, 'z': 0.862, 'confirmed': True, 'pose': 'walking_mirrored', 'footprint_xy': [[-22.451, -23.834], [-22.034, -24.19], [-21.339, -23.376], [-21.756, -23.02]]} | 0.87 | 5.3 | 6.0 | 8 |
| 2 | {'id': 'person_02', 'class': 'person', 'x': -10.238, 'y': -17.384, 'z': 0.214, 'confirmed': True, 'pose': 'lying_side_mirrored', 'footprint_xy': [[-10.959, -17.964], [-9.322, -17.509], [-9.518, -16.804], [-11.155, -17.259]]} | 1.00 | 13.7 | 16.7 | 20 |
| 3 | {'id': 'person_08', 'class': 'person', 'x': -14.731, 'y': -4.818, 'z': -1.122, 'confirmed': True, 'pose': 'lying_side', 'footprint_xy': [[-15.64, -4.864], [-15.323, -5.509], [-13.822, -4.771], [-14.139, -4.127]]} | 0.88 | 28.7 | 70.0 | 41 |
| 4 | {'id': 'person_09', 'class': 'person', 'x': -5.304, 'y': -12.889, 'z': 0.893, 'confirmed': True, 'pose': 'standing_pointing', 'footprint_xy': [[-5.812, -13.016], [-4.94, -13.265], [-4.796, -12.761], [-5.668, -12.512]]} | 0.77 | 13.1 | 58.7 | 19 |
| 5 | {'id': 'person_03', 'class': 'person', 'x': -4.545, 'y': -11.118, 'z': 0.501, 'confirmed': True, 'pose': 'sitting_mirrored', 'footprint_xy': [[-5.189, -11.197], [-4.069, -11.558], [-3.902, -11.039], [-5.021, -10.678]]} | 0.83 | 0.6 | 2.1 | 1 |
| 6 | {'id': 'person_04', 'class': 'person', 'x': -12.022, 'y': -10.682, 'z': 0.165, 'confirmed': True, 'pose': 'lying_back', 'footprint_xy': [[-11.843, -11.557], [-11.289, -11.193], [-12.2, -9.807], [-12.754, -10.171]]} | 0.88 | 11.5 | 34.3 | 16 |
| 7 | {'id': 'person_05', 'class': 'person', 'x': -17.683, 'y': -0.355, 'z': 0.436, 'confirmed': True, 'pose': 'sitting', 'footprint_xy': [[-18.215, -0.563], [-17.16, -0.586], [-17.15, -0.146], [-18.206, -0.123]]} | 0.77 | 17.3 | 62.4 | 25 |
| 8 | {'id': 'person_01', 'class': 'person', 'x': 10.709, 'y': 18.723, 'z': 0.989, 'confirmed': True, 'pose': 'standing_waving', 'footprint_xy': [[10.528, 18.443], [11.031, 18.639], [10.89, 19.003], [10.386, 18.807]]} | 0.73 | 60.9 | 81.2 | 87 |
| total | 8 people | | 151.0 | 331.4 | 216 |

Order by held_karp; walking time at 0.7 m/s; not reachable within 2.5 m: person_07, person_06, person_00.
