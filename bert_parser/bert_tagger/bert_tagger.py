from bert_parser.bert_tagger import BertWrapper, BertForLabelParsing


class BertTagger:
    model = None

    def load_trained_model(self, path):
        self.model = BertWrapper.load_serialized(path, BertForLabelParsing)

    def get_tags_for_list(self, li: list) -> dict:
        tagged = {}
        for unique in li:
            unique = str(unique)
            if unique not in tagged:
                tagged[unique] = self.predict_single_label(unique)[1]
        return tagged

    def predict_single_label(self, label):
        return label.split(), self.model.predict([label.split()])[0][0]

    def predict_batch_at_once(self, labels):
        return self.model.predict([label.split() for label in labels])[0]

    def predict_single_label_full(self, label):
        return label.split(), self.model.predict([label.split()])


