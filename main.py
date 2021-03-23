from processing import *
import threading
if __name__ == '__main__':
  """"#RFM_Futre()
   choose=int(input("Enter choose"))
   while True:
      if  choose==1:
           RFM_Segmentation()
           break
      elif choose==2:
           Figure_RFM_Log()
           break
      elif choose==3:
           Log_da_Tfd_Data()
      elif choose==4:
           Figure_Elow()
      elif choose==5:
           Customer_rfm_Kmean()
      elif choose==6:
         Figure_KMeanCluser()
      elif choose==7:
         Figure()
      else:
         Display()
   """
data=pd.read_csv("/home/rabit/PycharmProjects/pythonProject/Clusering.csv")
print(data['Cluster'].head())
