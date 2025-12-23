from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import Labels as lbl

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    lbl.x, lbl.y, test_size=0.2, random_state=42, stratify=lbl.y
)


model=LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')

# Train the model
model.fit(x_train, y_train)

#Predict on the test set
y_pred=model.predict(x_test)

#Evaluate the model

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

cm=confusion_matrix(y_test, y_pred, labels=np.unique(lbl.y))
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(lbl.y), yticklabels=np.unique(lbl.y))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


#Save the model
import joblib
model_filename='Number_Plate_Digit_Classifier.pkl'
joblib.dump(model, model_filename)
print(f"\nModel saved as {model_filename}")