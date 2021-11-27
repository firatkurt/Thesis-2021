import sys
sys.path.insert(0, r'C:\Users\FIRAT.KURT\Documents\Thesis_2021\src')
import importlib


class ModelManager:

    def GetModel(name):
        module = importlib.import_module('Model.' + name)
        class_ = getattr(module, name)
        return class_()
        