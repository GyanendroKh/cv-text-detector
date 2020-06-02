# Text Detector

## Installing Dependencies
```console
foo@bar:~$ python3 -m pip install -r requirements.txt --user
```

## Usage
```console
foo@bar:~$ python3 text_detection_video.py --video path/to/video.mp4 
```

## Help
```console
foo@bar:~$ python3 text_detection_video.py --help
Usage: text_detection_video.py [OPTIONS]

Options:
  --east PATH         Path to the East Text Detector Model.  [default:
                      frozen_east_text_detection.pb]

  --video PATH        Path to the video file.
  --duration INTEGER  Duration per frame (in milliseconds).  [default: 1500]
  --help              Show this message and exit.
```



