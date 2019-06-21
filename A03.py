from numpy import genfromtxt
from numpy import array 
from sklearn.model_selection import KFold

import numpy as np

'''
def calculateMeanAndVar(dataToTrain):
	print("Calculating mean and variance for ", dataToTrain)
	
	return 0
'''

def divideInClasses(dataToTrain):
	
	WindowClassList = []
	NonWindowClassList = []
	
	for i in range(len(dataToTrain)):
		checkData = dataToTrain[i][10]
		WindowClassList.append([])
		NonWindowClassList.append([])
		if(checkData >= 1 and checkData <=4):
			WindowClassList[i].append(dataToTrain[i])
		elif(checkData >=5 and checkData <=7):
			NonWindowClassList[i].append(dataToTrain[i])
			
	WindowClassList = [x for x in WindowClassList if x != []]
	NonWindowClassList = [y for y in NonWindowClassList if y !=[]]
		
	#print("Windowed Class is ", WindowClassList)
	#print("Non Windowed Class is ", NonWindowClassList)
	
	return WindowClassList, NonWindowClassList
	
	
def calculateMean(ClassA, ClassB):
	row1, row2 = 1,1
	Mean_ClassA = []
	Mean_ClassB = []
	listOfListA, listOfListB = [],[]
	count1,count2 = 0,0
	sum1,sum2 = 0,0
	
	while True:
		for col in range(len(ClassA)):
			data = ClassA[col][0][row1]
			sum1 +=data
			count1+=1
		Mean_ClassA.append(sum1/count1)
		listOfListA.append(Mean_ClassA)
		sum1 = 0
		count1=0
		if(row1 == 9):
			break
		row1+=1
	
	while True:
		for col2 in range(len(ClassB)):
			data = ClassB[col2][0][row2]
			sum2 +=data
			count2+=1
		Mean_ClassB.append(sum2/count2)
		listOfListB.append(Mean_ClassB)
		sum2 = 0
		count2=0
		if(row2 == 9):
			break
		row2+=1
		
	#print("list of sample mean for A is ", Mean_ClassA)	
	#print("list of sample mean for A is ", len(listOfListA))	
	#print("\n")
	#print("list of sample mean for B is ", Mean_ClassB)	
	
	#return listOfListA,listOfListB
	return Mean_ClassA,Mean_ClassB
	
def calculateVar(ClassOfData,MeanList):
	row = 1
	rowForMean = 0
	listOfSampleVar = []
	count = len(ClassOfData)
	
	sumVar = 0
	listOfCoVar = []
	
	while True:
		#print(rowForMean)
		for col in range(len(ClassOfData)):
			data = ClassOfData[col][0][row]
			#print("The data is ", data)
			sumVar += (data - MeanList[rowForMean])**2/count
			#print("End of loop")
		listOfSampleVar.append(sumVar)
		sumVar=0
		if(row == 9 or rowForMean==9):
			break
		row+=1
		rowForMean+=1
	
	#print("list of sample variance is ",listOfSampleVar)
	#print(listOfSampleVar[1])
	
	return listOfSampleVar
	
	
	
def ConvertListToMatrix(listOfVariance):
	listOfMatrix = []
	CovarMatrix = [[]]
	
	#print("THE LIST OF VARIANCE IS ",listOfVariance) # list of list
	#print(listOfVariance[0]) # one list
	#print(listOfVariance[0][0]) # element from one list
	
	countElemList = 0
	countRow = 0
	
	while True:
		#listToConvert = listOfVariance[countElemList]
		
		for i in range(len(listOfVariance)):
			#print(listToConvert[i])
			if(i == countRow):
				CovarMatrix[i].append(listOfVariance[i])
			else:
				CovarMatrix[i].append(0)
			
			CovarMatrix.append([])		
		#listOfMatrix.append(CovarMatrix)
		if(countElemList == len(listOfVariance) - 1):
			break
		else:
			countElemList +=1
		countRow+=1
		
	#for j in range(len(listOfMatrix)):
	#	listOfMatrix[j] = [x for x in listOfMatrix[j] if x != []]
	
	#for j in range(len(CovarMatrix)):
	CovarMatrix = [x for x in CovarMatrix if x != []]
		#print(CovarMatrix[j])
	
	#print(listOfMatrix)
	#print("The matrix is ",CovarMatrix)
	#print("Length of returning matrix for naive is ", len(CovarMatrix))
	return CovarMatrix
	
	
def calculateFullVar(ClassOfData,MeanList,VarList):
	CoVarMatrix = []
	IndexForMatrix = -1
	row = 1
	rowForMean = 0
	EndOfClassOfData = len(ClassOfData)-1
	elem = 0
	
	#print("VarList squared is ", np.outer(VarList,VarList))
	
	'''
	while True:
		Data_set = ClassOfData[elem][0]
		CoVarMatrix.append([])
		IndexForMatrix +=1
		for i in range(1,len(Data_set)-2,1):
			for j in range(len(MeanList)):
				sigmaIJ = (Data_set[i] - MeanList[j])*(Data_set[i] - MeanList[j])
				CoVarMatrix[IndexForMatrix].append(sigmaIJ)
		
		if(elem == EndOfClassOfData):
			break
		else:
			elem+=1
	
	print("The length of the covariacne matrix is ", len(CoVarMatrix))
	print(len(ClassOfData))
	CoVarMatrix = np.array(CoVarMatrix)
	#print("Length of returning matrix for optimal is ", len(CoVarMatrix))
	for c in range(len(CoVarMatrix)):
		print("The length of the element in matrix is ",len(CoVarMatrix[c]))
		print(CoVarMatrix[c])
	'''
	return np.outer(VarList,VarList)
	
def main():
	
	data_set = genfromtxt('Data.csv', delimiter=',')
	kfold = KFold(5,True)

	for train, test in kfold.split(data_set): # 5-fold cross validation scheme
		#print("train: "+ str(data_set[train]) + "test: " + str(data_set[test]))
		#pass
	
		dataToTrain = data_set[train]
		dataToTest  = data_set[test] 
		#print("dataToTest length is ", len(dataToTest[0]))
		
		#Dividing Training set into two classes: ClassA(Window class), ClassB(Non Window class)
		ClassA_list,ClassB_list = divideInClasses(dataToTrain)
		
		#Calculating sample mean for class A and class B		
		listOfMeanClassA, listOfMeanClassB = calculateMean(ClassA_list,ClassB_list)
		
		
		#Calculating variance Matrix for Class A and Class B for Naive Bayes Classifier
		listOfVarianceA = calculateVar(ClassA_list,listOfMeanClassA)
		listOfVarianceB = calculateVar(ClassB_list,listOfMeanClassB)
		
		
		
		#Calculating FULL Covariance Matrix for Class A and Class B for Optimal Bayes Classifier
		#NOTE: The matrix is already in Matrix form, NOT in 2d list
		#FullVarMatrixA = calculateFullVar(ClassA_list,listOfMeanClassA)
		FullVarMatrixA = calculateFullVar(ClassA_list,listOfMeanClassA,listOfVarianceA)
		FullVarMatrixB = calculateFullVar(ClassB_list,listOfMeanClassB,listOfVarianceB)
	
	
		#print("List of Variance for A is ", listOfVarianceA)
		#print("List of Variance for B is ", listOfVarianceB)
		
		
		#Converting listOfVar A and B to matrix
		listMatrixVarA = ConvertListToMatrix(listOfVarianceA)
		listMatrixVarB = ConvertListToMatrix(listOfVarianceB)
	
	
	
		#for i in range(len(listMatrixVarA)):
		listMatrixVarA = np.array(listMatrixVarA)
	
		#for j in range(len(listMatrixVarB)):
		listMatrixVarB = np.array(listMatrixVarB)
		
		
		#print("listMatrixVarA is ", listMatrixVarA)
		#print("listMatrixVarB is ", listMatrixVarB)
		
		#Perform Naive Bayes Classifier Equation and output result
	
		#Determining the ln of determinant of class A and class B
		LnOfdetA = np.linalg.det(listMatrixVarA) 
		LnOfdetB = np.linalg.det(listMatrixVarB) 
	
		#print("Determinant of class A is ", LnOfdetA)
		#print("Determinant of class B is ", LnOfdetB)
	
		#Determining the Mahalanobis distance function for class A and class B
		#loop through the testing data and and plug in the TRANSPOSE of test vector and calculate 
	
		### THIS IS FOR inverse of NAIVE BAYES VARIANCE ##
		VarianceMinusPowA = np.linalg.pinv(listMatrixVarA)
		#print("variance of matrix A is ", VarianceMinusPowA)
		VarianceMinusPowB = np.linalg.pinv(listMatrixVarB)
		#print("variance of matrix B is ", VarianceMinusPowB)
		
		
		## THIS IS FOR INVERSE OF OPTIMAL BAYES VARIANCE ##
		FullVarInvA = np.linalg.pinv(FullVarMatrixA)
		#print("Variance of matrix A in FULL VARIANCE IS ",FullVarInvA)
		FullVarInvB = np.linalg.pinv(FullVarMatrixB)
		
		
		ActualTrueValue = []
		TestingValue_Naive = []
		TestingValue_Optimal = []
		countTrue_Naive = 0
		countTrue_Optimal = 0
		########################################## TESTING FOR NAIVE BAYES CLASSIFIER###############################
		for i in range(len(dataToTest)):
			#print(i)
			testingData = dataToTest[i] # x-vector in Mahalanobis distance function 
			DataClassify = dataToTest[i]
			
			testingData = list(testingData)
			del testingData[0]
			del testingData[9]
				
			testingData= np.array(testingData)
		
			
			xMinusMeanA = np.subtract(testingData,listOfMeanClassA)
			xMinusMeanB = np.subtract(testingData,listOfMeanClassB)
			
			xMinusMeanT_A = np.transpose(xMinusMeanA)
			xMinusMeanT_B = np.transpose(xMinusMeanB)
			
			
			D_FunctionA_Naive = np.dot(np.dot(xMinusMeanT_A, VarianceMinusPowA),xMinusMeanA) 
			D_FunctionB_Naive = np.dot(np.dot(xMinusMeanT_B, VarianceMinusPowB),xMinusMeanB) 
			
			D_FunctionA_Optimal = np.dot(np.dot(xMinusMeanT_A, FullVarInvA),xMinusMeanA)
			D_FunctionB_Optimal = np.dot(np.dot(xMinusMeanT_B, FullVarInvB),xMinusMeanB)
			
			
			c_value_Naive = LnOfdetB - LnOfdetA + D_FunctionB_Naive - D_FunctionA_Naive
			#print("C_Vlaue for naive is ", c_value_Naive)
			
			c_value_Optimal = LnOfdetB - LnOfdetA + D_FunctionB_Optimal - D_FunctionA_Optimal
			#print("C_Vlaue for optimal is ", c_value_Optimal)
			
			TestingValue_Naive.append(c_value_Naive)
			TestingValue_Optimal.append(c_value_Optimal)
			ActualTrueValue.append(DataClassify[10])
		
		#print("Len of c_value in NAIVE", len(TestingValue_Naive))
		#print("Len of c_value in OPTIMAL", len(TestingValue_Optimal))
		#print("Len of true value", len(ActualTrueValue))
			
		
		for element in range(len(TestingValue_Naive)):
			if(TestingValue_Naive[element] < 0):
				if(ActualTrueValue[element] >=5 and ActualTrueValue[element] <= 7):
					countTrue_Naive +=1
			
			elif(TestingValue_Naive[element] > 0):
				if(ActualTrueValue[element] >=1 and ActualTrueValue[element] <= 4):
					countTrue_Naive+=1
					
		for element in range(len(TestingValue_Optimal)):
			if(TestingValue_Optimal[element] < 0):
				if(ActualTrueValue[element] >=5 and ActualTrueValue[element] <= 7):
					countTrue_Optimal +=1
			
			elif(TestingValue_Optimal[element] > 0):
				if(ActualTrueValue[element] >=1 and ActualTrueValue[element] <= 4):
					countTrue_Optimal+=1
		
		print("The Accuracy Value for Naive Bayes Classifier is ", (countTrue_Naive/len(ActualTrueValue)) * 100)
		print("The Accuracy Value for Optimal Bayes Classifier is ", (countTrue_Optimal/len(ActualTrueValue)) * 100)
		print("\n")
	
	
	
	
	
			
	
	
	
	
		
	
	
	
	
main()




















