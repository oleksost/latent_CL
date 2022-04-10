from dataclasses import dataclass
from simple_parsing import mutable_field     
from Models.classifiers import Classifier_BiT, Classifier_nn, RandomForest_Classifier, KNeighbors_Classifier, LogisticRegression_Classifier, CLIPZeroShotClassifier, NMC_Classifier, WeightNorm_Classifier, SLDA_Classifier

@dataclass
class Classifier_options:
    CLS_BiT: Classifier_BiT.Options = mutable_field(Classifier_BiT.Options)
    CLS_NN: Classifier_nn.Options = mutable_field(Classifier_nn.Options)
    CLS_RF: RandomForest_Classifier.Options = mutable_field(RandomForest_Classifier.Options)
    CLS_LogReg: LogisticRegression_Classifier.Options = mutable_field(LogisticRegression_Classifier.Options)
    CLP_0: CLIPZeroShotClassifier.Options = mutable_field(CLIPZeroShotClassifier.Options)
    CLS_KNN: KNeighbors_Classifier.Options = mutable_field(KNeighbors_Classifier.Options)
    CLS_NMC: NMC_Classifier.Options = mutable_field(NMC_Classifier.Options)
    CLS_WeightNorm: WeightNorm_Classifier.Options = mutable_field(WeightNorm_Classifier.Options)
    CLS_SLDA: SLDA_Classifier.Options = mutable_field(SLDA_Classifier.Options)
    CLS_NMC: NMC_Classifier.Options = mutable_field(NMC_Classifier.Options)
