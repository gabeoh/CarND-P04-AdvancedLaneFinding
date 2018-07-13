# CarND-P04-AdvancedLaneFinding

CarND-P04-AdvancedLaneFinding implements a pipeline to detect lane lines
from images and video streams. 

## File Structure
### Project Requirements
- **[py-src/](py-src/)** - Contains Python source codes that implement the
    pipeline
- **[output_images/](output_images/)** - Contains resulting output images and
    videos for each pipeline step
- **[writeup_report.md](writeup_report.md)** - Project write-up report

### Additional Files
- **[results/](results/)** - Project outputs such as pickle and execution
    log files
- **[camera_cal/](camera_cal/)** - 
- **[test_images/](test_images/)** - 
- **[test_videos/](test_videos/)** - 


## Getting Started
### [Download ZIP](https://github.com/gabeoh/CarND-P04-AdvancedLaneFinding/archive/master.zip) or Git Clone
```
git clone https://github.com/gabeoh/CarND-P04-AdvancedLaneFinding.git
```

### Setup Environment

You can set up the environment following
[CarND-Term1-Starter-Kit - Miniconda](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md).
This will install following packages required to run this application.

- Miniconda
- Python
- Jupyter Notebook

### Download Simulator
- [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
- [MacOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)
- [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)


### Usage

#### Run Camera Calibration
Using calibration images under `camera_cal`, compute camera matrix and
distortion coefficients.  The resulting data is stored as a pickle file
`results/camera_cal.p`.

```
$ cd py-src

$ python p05_00_camera_calibration.py
```

#### Run Lane Detection on Test Images
```
$ cd py-src

$ python p05_lane_detection_main.py --image
or
$ python p05_lane_detection_main.py -i
```

You can also run only specific steps.  For example, run only step 3
perspective transform and step 4 lane line identification.
```
$ python p05_lane_detection_main.py -i 3 4
```

More information on running option can be found by running:
```
$ python p05_lane_detection_main.py -h
```

#### Run Lane Detection on Test Videos
```
$ cd py-src

$ python p05_lane_detection_main.py --video
or
$ python p05_lane_detection_main.py -v
```

You can also run on only specific video files.  For example, run the pipeline
only on project video.
```
$ python p05_lane_detection_main.py -v -f project_video.mp4
```

More information on running option can be found by running:
```
$ python p05_lane_detection_main.py -h
```


## License
Licensed under [MIT](LICENSE) License.
