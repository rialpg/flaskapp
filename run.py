import argparse
import app.CONFIG as CONFIG
from app import app


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Program input flags
    # These flags can be set in the command line interface when running the command
    # sample command is 
    # $ python run.py --flag1=value --flag2=value ...

    parser.add_argument('--threshold', type=float, required=False, default=0.7, 
        help='Probability threshold above which a person is to be classified, else unknown, default=%(default)s')

    parser.add_argument('--training_images', type=int, required=False, default=10, 
        help='# of images required to train the classifier on, greater number brings better accuracy, default=%(default)s')

    parser.add_argument('--gpu', default=False, action="store_true",
        help = "Specify whether to use GPU, default=%(default)s")

    parser.add_argument('--resize_scale', type=float, required=False, default=1.0,
        help = "Resize the input video, e.g. 0.75 will resize to 1/4th of the video. default is 1 means No resize")
    
    # parsing the flags into a variable called args
    args = parser.parse_args()

    CONFIG.TRAINING_IMAGES = args.training_images
    CONFIG.CONFIDENCE_THRESHOLD = args.threshold
    CONFIG.GPU = args.gpu
    CONFIG.RESIZE_SCALE = args.resize_scale

    # run the server
    app.run(host='0.0.0.0', debug=True)
