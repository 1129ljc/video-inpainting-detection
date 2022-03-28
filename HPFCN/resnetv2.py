import torch
import timm
import sys
import traceback

model_list = timm.list_models('*resnetv2_50*')
for i in model_list:
    print(i)


def TraceStack():
    print("--------------------")
    frame = sys._getframe(1)
    while frame:
        print(frame.f_code.co_name)
        print(frame.f_code.co_filename)
        print(frame.f_lineno)
        frame = frame.f_back


def get_resnetv2():
    model = timm.create_model('resnetv2_50', pretrained=False)
    print(model)
    traceback.print_stack()

if __name__ == '__main__':
    get_resnetv2()


