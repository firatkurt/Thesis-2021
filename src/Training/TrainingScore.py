class TrainingScore:
    def __init__(self):
        self.accuracy_score = None
        self.precision_score = None
        self.recall_score = None
        self.confusion_matrix = None

    def __str__(self):
        list = [str(score) if score is not None else '' for score in [self.accuracy_score,self.precision_score,self.recall_score, self.confusion_matrix]]
        return ','.join(list)
