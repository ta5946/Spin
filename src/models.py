class ConstantClassifier:
    def __init__(self, constant):
        self.constant = constant

    def predict(self, out1, out2):
        return self.constant
