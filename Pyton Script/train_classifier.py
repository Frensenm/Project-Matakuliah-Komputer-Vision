import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Convert labels to integer
labels = np.asarray([int(label) for label in labels])

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# Define XGBoost model
model = xgb.XGBClassifier(
    n_estimators=300,       # Jumlah pohon keputusan
    max_depth=10,           # Kedalaman maksimal pohon
    learning_rate=0.05,     # Kecepatan belajar (lebih kecil lebih baik)
    subsample=0.8,          # Menggunakan 80% data untuk setiap pohon
    colsample_bytree=0.8,   # Fitur yang dipilih untuk setiap pohon
    objective='multi:softmax',  # Softmax untuk klasifikasi multi-kelas
    num_class=len(set(labels)),  # Jumlah kelas sesuai dataset
    eval_metric="mlogloss",
    early_stopping_rounds=10,  # Hentikan training jika tidak ada peningkatan setelah 10 epoch
    use_label_encoder=False
)

# Train model
model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=True)

# Predict
y_predict = model.predict(x_test)

# Accuracy
score = accuracy_score(y_test, y_predict)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Classification report & Confusion Matrix
print("\nClassification Report:\n", classification_report(y_test, y_predict))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_predict))

# Buat confusion matrix
cm = confusion_matrix(y_test, y_predict)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(labels), 
            yticklabels=np.unique(labels))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

xgb.plot_importance(model)
plt.show()


# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
