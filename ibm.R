setwd("C:\\Users\\Patricia\\Desktop\\FIA\\TCC\\ibm-hr-analytics-attrition-dataset")

library(car)
library(readxl)
library(psych)
library(tidyr)
library(dplyr)
library(ggplot2)
library(caret)
library(e1071)
library(rpart) ##árvore de decisão
library(rpart.plot) ##árvore de decisão
library(expss)
library(corrplot)

base<-read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv", sep = ",")
df<-read.csv("WA_Fn-UseC_-HR-Employee-Attrition-pearson.csv", sep = ";")
head(base)
colnames(base)

###### correlação de pearson

m<-cor(df)
corrplot(m, method = "circle")

pairs.panels(df)


#Attrition - Employee leaving the company (0=no, 1=yes) 
base$Attrition <- as.character(base$Attrition)
base$Attrition[base$Attrition == "No"] <- "0"
base$Attrition[base$Attrition == "Yes"] <- "1"
freq <- table(base$Attrition) #calculando as frequencias

#Tabela dinâmica com os percentuais. Melhor visualização que as aulas anteriores
cro_rpct(base$WorkLifeBalance, base$Attrition)


base_char <-base[,-1] #Tira a primeira coluna (id)
chrs <- sapply(base_char, is.character) #verifica as colunas qualitativas
chrCols <- names(base_char[, chrs]) #mantém em uma outra base (chrCols) só as var. qualitativas

#Macro para calcular a tabela dinâmica para todas as variáveis
for (i in chrCols){
  print(i)
  print(table(base_char[i]), useNA = "always")
  print(cro_rpct(base_char[i], base_char$Attrition))
}


###Gráfico de pizza
pie(freq, main="Attrition", labels =c("83,88%", "16,12%"), col = c(4,2))
legend("topright", fill = c(4,2), legend = c("No leaving", "Leaving"))

ggplot(base, aes(x=factor(1), fill=Attrition))+
  geom_bar(width = 1)+
  coord_polar("y")

#boxplot da var. qualitaiva x resposta
boxplot(base$ï..Age~base$Attrition)

###Correlação de pearson - correlação entre variaveis
vars <- names(base)[1:4]
cor(base[,vars])

##Análise bivariada - gráfico de dispersão
plot(base$ï..Age, base$MonthlyIncome, pch=16, main = "Gráfico Dispersão",
     xlab="Age", ylab="MonthlyIncome", ylim=c(500,22000))

#Cria o box plot das var. quantitativas x resposta. Forma mais bonita.
base %>%
  select(ï..Age, DailyRate, DistanceFromHome, 
          NumCompaniesWorked, Attrition) %>%
  gather(Measure, Value, -Attrition) %>%
  ggplot(aes(x = factor(Attrition)
             , y = Value)) +
  geom_boxplot() +
  facet_wrap(~Measure
             , scales = "free_y")

base %>%
  select(PercentSalaryHike, YearsAtCompany, 
         YearsInCurrentRole, YearsSinceLastPromotion, Attrition) %>%
  gather(Measure, Value, -Attrition) %>%
  ggplot(aes(x = factor(Attrition)
             , y = Value)) +
  geom_boxplot() +
  facet_wrap(~Measure
             , scales = "free_y")


base %>%
  select( MonthlyRate, TotalWorkingYears, 
         MonthlyIncome, TrainingTimesLastYear, Attrition) %>%
  gather(Measure, Value, -Attrition) %>%
  ggplot(aes(x = factor(Attrition)
             , y = Value)) +
  geom_boxplot() +
  facet_wrap(~Measure
             , scales = "free_y")

base %>%
  select( TotalWorkingYears, 
          TrainingTimesLastYear, Attrition) %>%
  gather(Measure, Value, -Attrition) %>%
  ggplot(aes(x = factor(Attrition)
             , y = Value)) +
  geom_boxplot() +
  facet_wrap(~Measure
             , scales = "free_y")


base %>%
  select( StockOptionLevel, YearsWithCurrManager,
          HourlyRate, Attrition) %>%
  gather(Measure, Value, -Attrition) %>%
  ggplot(aes(x = factor(Attrition)
             , y = Value)) +
  geom_boxplot() +
  facet_wrap(~Measure
             , scales = "free_y")

summary(base)
#painel de todas as variáveis quantitativas - correlação
basepanels<-read.csv("WA_Fn-UseC_-HR-Employee-Attrition-painels.csv", sep = ";")
pairs.panels(basepanels[1:15])
pairs.panels(basepanels[1:8])
pairs.panels(basepanels[8:15])

cor(base)

##Transformar as variáveis categóricas

#BusinessTravel - (1=No Travel, 2=Travel Frequently, 3=Tavel Rarely) 
base$BusinessTravel <- as.character(base$BusinessTravel)
base$BusinessTravel[base$BusinessTravel == "Non-Travel"] <- "1"
base$BusinessTravel[base$BusinessTravel == "Travel_Frequently"] <- "2"
base$BusinessTravel[base$BusinessTravel == "Travel_Rarely"] <- "3"
table(base$BusinessTravel) #calculando as frequencias

#Department - (1=Human Resources, 2=Research & Development, 3=Sales) 
base$Department <- as.character(base$Department)
base$Department[base$Department == "Human Resources"] <- "1"
base$Department[base$Department == "Research & Development"] <- "2"
base$Department[base$Department == "Sales"] <- "3"
table(base$Department) #calculando as frequencias

#Education Field - (1=Human Resources, 2=Life Sciences, 3=Marketing, 4=Medical, 5=Other) 
base$EducationField <- as.character(base$EducationField)
base$EducationField[base$EducationField == "Human Resources"] <- "1"
base$EducationField[base$EducationField == "Life Sciences"] <- "2"
base$EducationField[base$EducationField == "Marketing"] <- "3"
base$EducationField[base$EducationField == "Medical"] <- "4"
base$EducationField[base$EducationField == "Other"] <- "5"
base$EducationField[base$EducationField == "Technical Degree"] <- "6"

table(base$EducationField) #calculando as frequencias

#Gender - (1=Female, 2=Male) 
base$Gender <- as.character(base$Gender)
base$Gender[base$Gender == "Female"] <- "1"
base$Gender[base$Gender == "Male"] <- "2"

table(base$Gender) #calculando as frequencias


#job Role - (1=Healthcare Representative, 2=Human Resources, 3=Laboratory Technician, 4=Manager, 
# 5=Manufacturing Director, 6=Research Director, 7=Research Scientist, 8=Sales Executive, 9=Sales Representative) 
base$JobRole <- as.character(base$JobRole)
base$JobRole[base$JobRole == "Healthcare Representative"] <- "1"
base$JobRole[base$JobRole == "Human Resources"] <- "2"
base$JobRole[base$JobRole == "Laboratory Technician"] <- "3"
base$JobRole[base$JobRole == "Manager"] <- "4"
base$JobRole[base$JobRole == "Manufacturing Director"] <- "5"
base$JobRole[base$JobRole == "Research Director"] <- "6"
base$JobRole[base$JobRole == "Research Scientist"] <- "7"
base$JobRole[base$JobRole == "Sales Executive"] <- "8"
base$JobRole[base$JobRole == "Sales Representative"] <- "9"

table(base$JobRole) #calculando as frequencias

#mARITAL STATUS - (1=Divorced, 2=Married, 3=Single) 
base$MaritalStatus <- as.character(base$MaritalStatus)
base$MaritalStatus[base$MaritalStatus == "Divorced"] <- "1"
base$MaritalStatus[base$MaritalStatus == "Married"] <- "2"
base$MaritalStatus[base$MaritalStatus == "Single"] <- "3"

table(base$MaritalStatus) #calculando as frequencias

#over18 - (1=Y, 2=N) 
base$Over18 <- as.character(base$Over18)
base$Over18[base$Over18 == "Y"] <- "1"
base$Over18[base$Over18 == "N"] <- "2"

table(base$Over18) #calculando as frequencias

#over18 - (1=Y, 2=N) 
base$OverTime <- as.character(base$OverTime)
base$OverTime[base$OverTime == "Yes"] <- "1"
base$OverTime[base$OverTime == "No"] <- "2"

table(base$OverTime) #calculando as frequencias

#Tranbsforma em caracter
base$EnvironmentSatisfaction <- as.character(base$EnvironmentSatisfaction)
base$Education <- as.character(base$Education)
base$JobInvolvement <- as.character(base$JobInvolvement)
base$JobLevel <- as.character(base$JobLevel)
base$JobSatisfaction <- as.character(base$JobSatisfaction)
base$PerformanceRating <- as.character(base$PerformanceRating)
base$RelationshipSatisfaction <- as.character(base$RelationshipSatisfaction)
base$WorkLifeBalance <- as.character(base$WorkLifeBalance)


#Balancear a base de dados - 50% e 50%

# Coloca os leaving em uma base de dados
leaving <- subset(base, base$Attrition==1)

#Selecionar, aleatoriamente, 237 observações dos não leaving
set.seed(123) #para obter sempre a mesma amostra

no_leaving <- subset(base, base$Attrition==0)
dt = sort(sample(nrow(no_leaving), 237))
no_leaving<-no_leaving[dt,]

# Junta as duas bases
base_balanceada = rbind(leaving, no_leaving)
table(base_balanceada$Attrition)

##transformar variável resposta em numérico
base_balanceada$Attrition <- as.numeric(as.character(base_balanceada$Attrition))

#Divide em base de treino e teste
dt = sort(sample(nrow(base_balanceada), nrow(base_balanceada)*.7))
train<-base_balanceada[dt,]
test<-base_balanceada[-dt,] 

####padronização das bases
train = train %>%
  mutate_if(is.numeric, scale)

test = test %>%
  mutate_if(is.numeric, scale)

##transformar variável resposta em numérico
train$Attrition <- as.numeric(as.character(train$Attrition))

#Modelo GLM - regressao logistica
full.model <- glm(Attrition ~
                    ï..Age + BusinessTravel +	DailyRate +	Department+	DistanceFromHome
                  +Education+	EducationField
                  +EnvironmentSatisfaction	+Gender
                  +HourlyRate	+JobInvolvement	+JobLevel
                  +JobRole	+JobSatisfaction	+MaritalStatus
                  +MonthlyIncome	+MonthlyRate	+NumCompaniesWorked
                  +OverTime	+PercentSalaryHike
                  +PerformanceRating	+RelationshipSatisfaction
                  +StockOptionLevel	+TotalWorkingYears	+TrainingTimesLastYear
                  +WorkLifeBalance	+YearsAtCompany	+YearsInCurrentRole
                  +YearsSinceLastPromotion	+YearsWithCurrManager,
                  family=binomial(link='logit'),data=train)
summary(full.model)

####modelagem com seleção de variáveis backward

step(full.model, direction = "backward")

####retirar backward +	DailyRate +	Department, Education +HourlyRate
###+MonthlyIncome +PercentSalaryHike   +PerformanceRating	
###+StockOptionLevel	+TotalWorkingYears	 +WorkLifeBalance	

full.model_bac <- glm(Attrition ~
                    ï..Age + BusinessTravel +	DistanceFromHome
                  +	EducationField
                  +EnvironmentSatisfaction	+Gender
                  +JobInvolvement	+JobLevel
                  +JobRole	+JobSatisfaction	+MaritalStatus
                  +MonthlyRate	+NumCompaniesWorked
                  +OverTime	+RelationshipSatisfaction
                  +TrainingTimesLastYear
                  +YearsAtCompany	+YearsInCurrentRole
                  +YearsSinceLastPromotion	+YearsWithCurrManager,
                  family=binomial(link='logit'),data=train)
summary(full.model_bac)

pred = predict(full.model_bac, train, type = "response")
finaldata = cbind(train, pred) #colocar a base de dados

describeBy(finaldata$pred , finaldata$Attrition) #media das probb por resposta
### perguntar como analisar esse describeby

#calcular as variaveis finais
finaldata$response <- as.factor(ifelse(finaldata$pred>0.7, 1, 0)) 

#Matriz de confusão
confusionMatrix(table(finaldata$response,finaldata$Attrition))

#### replicar na base de teste o predict

pred = predict(full.model_15,test, type = "response")
finaldata = cbind(test, pred) #colocar a base de dados
finaldata$response <- as.factor(ifelse(finaldata$pred>0.7, 1, 0)) 
confusionMatrix(table(finaldata$response, finaldata$Attrition))

###Árvore de decisão

arvore<- rpart(Attrition ~
                 BusinessTravel +DistanceFromHome
               +EnvironmentSatisfaction	+Gender
               +JobInvolvement
               +JobSatisfaction	+MaritalStatus
               +NumCompaniesWorked
               +OverTime
               +RelationshipSatisfaction
               +TotalWorkingYears	+TrainingTimesLastYear
               +YearsAtCompany	+YearsInCurrentRole
               +YearsSinceLastPromotion	+YearsWithCurrManager, method="class",data=train, cp) 
plot(arvore, uniform=TRUE,
     main="Árvore de Classificação para o Clicked")
text(arvore, use.n=TRUE, all=TRUE, cex=.8)

rpart.plot(arvore, extra = 106)


arvore <- rpart(Attrition~., data = train, method = 'class', control=rpart.control(minsplit=60, cp=0.001) )
rpart.plot(arvore, extra = 106)


pred = predict(arvore, type="class",newdata=test)
finaldata = cbind(test, pred)
confusionMatrix(finaldata$pred,finaldata$Churn2 )


