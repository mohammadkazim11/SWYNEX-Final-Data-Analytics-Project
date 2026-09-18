import pandas as pd
# load data set
df=pd.read_csv("train.csv")

print("Dataset:")

# display the first 5 rows of the dataset
print(df.head())

# missing values
print("\nMissing values:")
print(df.isnull().sum())

# duplicate values
print("\nDuplicate values:")
print(df.duplicated().sum())

#data types
print("\nData types:")
print(df.dtypes)

# check unique values
print("\nSex values:")
print(df["Sex"].unique())

print("\nEmbarked values:")
print(df["Embarked"].unique())


# handle missing values
df["Age"]= df["Age"].fillna(df["Age"].median())

df["Embarked"]= df["Embarked"].fillna(df["Embarked"].mode()[0])

print("\n Missing values after cleaning:")
print(df.isnull().sum())

# remove cabin column because it has too many missing values 
df=df.drop("Cabin",axis=1)

print("\nMissing values after dropping Cabin column:")
print(df.isnull().sum())

df.to_csv("cleaned_titanic.csv",index=False)
print("\n Cleaned dataset saved  successfully!") 