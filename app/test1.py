import sys
import pandas as pd

# from util.Model4 import train_model_4
# from util import train_model
#from util.Model3 import train_model_3
# from util.ModelPredictor import predict
from util.ModelPredictor import predict_simi

sys.path.append(".")

# train_model()
# train_model_4()
#
# data1 = pd.read_csv('./../resources/similarity1.csv', sep='\t')
# data2 = pd.read_csv('./../resources/similarity2.csv', sep='\t')
# data3 = pd.read_csv('./../resources/similarity3.csv', sep='\t')
# data4 = pd.read_csv('./../resources/similarity4.csv', sep='\t')
# data5 = pd.read_csv('./../resources/similarity5.csv', sep='\t')
# data6 = pd.read_csv('./../resources/similarity6.csv', sep='\t')
# data7 = pd.read_csv('./../resources/similarity7.csv', sep='\t')
# data8 = pd.read_csv('./../resources/similarity8.csv', sep='\t')
#
# data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8])
# data = data.reset_index()
# data = data[data.t1 != ""]
# data = data[data.t2 != ""]
# data = data[data.t1 != ""]
# data = data[data.t2 != "[]"]
#
# data.to_csv('./../resources/similarity.csv', sep='\t')

#train_model_3()

predict_simi()
# train_model()

# predict()
