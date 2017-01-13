__author__ = 'diego'


class LinearDecay:

    startValue = 0.0
    endValue = 0.0
    maxIteration = 0
    decay = False
    def __init__(self,startValue = float, endValue = float, maxIteration = int, decay = bool):
        self.endValue = endValue
        self.startValue = startValue
        self.maxIteration = maxIteration
        self.decay = decay


    def apply(self, curruentIteration):
        mul = 1
        if self.decay:
			mul = -1
        return self.startValue + mul * curruentIteration * self.endValue/(self.maxIteration)