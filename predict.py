import argparse
import numpy as np

from helper_functions import go_go_categories, go_go_load_checkpoint, go_go_predict, go_go_print

arg_parse = argparse.ArgumentParser(description='predict.py')
arg_parse.add_argument('--image_dir', default='./flowers/valid/18/image_04252.jpg', nargs='*', action="store", type = str)
arg_parse.add_argument('--image_category', default='18', nargs='*', action="store", type = str)
arg_parse.add_argument('--topk', default=10, nargs='*', action="store", type = int)
arg_parse.add_argument('--checkpoint', default='./checkpoint.pth', nargs='*', action="store", type = str)
args = arg_parse.parse_args()
go_go_print('Loading Model from checkpoint "%s"' % args.checkpoint)

try:
    model, criterion, optimizer, model_settings = go_go_load_checkpoint(args.checkpoint)
except Exception as e:
    print(e)
    print('No checkpoint found. Please train the model before attempting to predict.')

predictions, predictedCategories, flower_categories = go_go_predict(args.image_dir, model, args.topk)
predictions = predictions * 100
print("Actual Flower Name: '{}'".format(flower_categories[args.image_category]))

i = 0
while i < len(predictions):
    print("Rank: {} | Flower Name: '{}' | Model Prediction: {:.2f}%".format(i + 1, predictedCategories[i], predictions[i]))
    i += 1
go_go_print('May the force be with you :)')
