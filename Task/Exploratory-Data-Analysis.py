import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("cleaned_titanic.csv")

# Basic information
print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSurvival Count:")
print(df["Survived"].value_counts())



# ---------------- EDA VISUALIZATIONS ----------------

fig, axes = plt.subplots(3, 2, figsize=(14, 16))

# 1. Survival Count
survival_count = df["Survived"].value_counts().sort_index()

axes[0, 0].bar(["Not Survived", "Survived"], survival_count.values)
axes[0, 0].set_title("Survival Count", fontsize=14, fontweight="bold")
axes[0, 0].set_ylabel("Number of Passengers")
axes[0, 0].grid(axis="y", linestyle="--", alpha=0.3)

for i, value in enumerate(survival_count.values):
    axes[0, 0].text(i, value + 10, str(value), ha="center")


# 2. Survival Rate by Gender
gender_rate = df.groupby("Sex")["Survived"].mean() * 100

axes[0, 1].bar(gender_rate.index, gender_rate.values)
axes[0, 1].set_title("Survival Rate by Gender", fontsize=14, fontweight="bold")
axes[0, 1].set_ylabel("Survival Rate (%)")
axes[0, 1].set_ylim(0, 100)
axes[0, 1].grid(axis="y", linestyle="--", alpha=0.3)

for i, value in enumerate(gender_rate.values):
    axes[0, 1].text(i, value + 2, f"{value:.1f}%", ha="center")


# 3. Age Distribution
mean_age = df["Age"].mean()

axes[1, 0].hist(
    df["Age"].dropna(),
    bins=20,
    edgecolor="black",
    alpha=0.8
)

axes[1, 0].axvline(
    mean_age,
    linestyle="--",
    linewidth=2,
    label=f"Mean Age: {mean_age:.1f}"
)

axes[1, 0].set_title("Age Distribution", fontsize=14, fontweight="bold")
axes[1, 0].set_xlabel("Age")
axes[1, 0].set_ylabel("Number of Passengers")
axes[1, 0].grid(axis="y", linestyle="--", alpha=0.3)
axes[1, 0].legend()


# 4. Passenger Class Distribution
class_count = df["Pclass"].value_counts().sort_index()

axes[1, 1].bar(class_count.index.astype(str), class_count.values)
axes[1, 1].set_title("Passenger Class Distribution", fontsize=14, fontweight="bold")
axes[1, 1].set_xlabel("Passenger Class")
axes[1, 1].set_ylabel("Number of Passengers")
axes[1, 1].grid(axis="y", linestyle="--", alpha=0.3)

for i, value in enumerate(class_count.values):
    axes[1, 1].text(i, value + 10, str(value), ha="center")


# 5. Fare Distribution
axes[2, 0].hist(
    df["Fare"].dropna(),
    bins=30,
    edgecolor="black",
    alpha=0.8
)

axes[2, 0].set_title("Fare Distribution", fontsize=14, fontweight="bold")
axes[2, 0].set_xlabel("Fare")
axes[2, 0].set_ylabel("Frequency")
axes[2, 0].grid(axis="y", linestyle="--", alpha=0.3)


# Empty 6th position
axes[2, 1].axis("off")

plt.suptitle(
    "Titanic Dataset - Exploratory Data Analysis",
    fontsize=20,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig(
    "EDA_visualizations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


#  KEY INSIGHTS 

print("\n========== KEY INSIGHTS ==========")

# 1. Overall survival
overall_survival = df["Survived"].mean() * 100
print(f"1. Overall, {overall_survival:.1f}% of passengers survived.")

# 2. Gender survival
gender_survival = df.groupby("Sex")["Survived"].mean() * 100
print(
    f"2. Female survival rate was {gender_survival['female']:.1f}%, "
    f"while male survival rate was {gender_survival['male']:.1f}%."
)

# 3. Passenger class survival
class_survival = df.groupby("Pclass")["Survived"].mean() * 100
print(
    f"3. Survival rate was {class_survival[1]:.1f}% in 1st class, "
    f"{class_survival[2]:.1f}% in 2nd class, and "
    f"{class_survival[3]:.1f}% in 3rd class."
)

# 4. Average age
print(f"4. The average age of passengers was {df['Age'].mean():.1f} years.")

# 5. Average fare
print(f"5. The average passenger fare was {df['Fare'].mean():.2f}.")