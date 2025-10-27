# ===============================================================
# Title: Anomaly Detection in Network Traffic using AI/ML (NSL-KDD)
# Dataset: NSL-KDD (Train/Test CSV from Hugging Face)
# Author: Your Team Name
# ===============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras import models, layers
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ===============================================================
# Step 1: Load Dataset
# ===============================================================
print("📥 Loading NSL-KDD dataset...")
df_train = pd.read_csv("KDDTrain+.csv")
df_test = pd.read_csv("KDDTest+.csv")
df = pd.concat([df_train, df_test], ignore_index=True)

print("✅ Dataset Loaded Successfully!")
print("Shape:", df.shape)
print(df.head())

# ===============================================================
# Step 2: Simplify Labels (normal vs attack)
# ===============================================================
df['label'] = df['label'].apply(lambda x: 'normal' if x.strip().lower() == 'normal' else 'attack')
print("\nLabel Distribution:\n", df['label'].value_counts())

# ===============================================================
# Step 3: Encode Categorical Columns
# ===============================================================
categorical_cols = ['protocol_type', 'service', 'flag']
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# ===============================================================
# Step 4: Split Features and Labels + Scaling
# ===============================================================
X = df.drop(['label'], axis=1)
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# ===============================================================
# Step 5: Supervised ML Models
# ===============================================================
models_list = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []

for name, model in models_list.items():
    print(f"\n⚙ Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))
    
    print(f"\n===== {name} Results =====")
    print("Accuracy:", round(acc*100, 2), "%")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# ===============================================================
# Step 6: Compare Model Performance
# ===============================================================
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)
print("\n🏆 Model Comparison:\n", results_df)

# ===============================================================
# Step 7: Unsupervised Learning (Isolation Forest)
# ===============================================================
print("\n🧩 Running Isolation Forest...")
iso_forest = IsolationForest(contamination=0.2, random_state=42)
iso_pred = iso_forest.fit_predict(X_scaled)
iso_pred = np.where(iso_pred == -1, 'attack', 'normal')

print("\n===== Isolation Forest Results =====")
print(classification_report(y, iso_pred))

# ===============================================================
# Step 8: Deep Learning Autoencoder (for anomaly detection)
# ===============================================================
print("\n🤖 Training Autoencoder...")
input_dim = X_scaled.shape[1]
autoencoder = models.Sequential([
    layers.Dense(64, activation='relu', input_dim=input_dim),
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')
])
autoencoder.compile(optimizer='adam', loss='mse')

# Train only on normal data
X_normal = X_scaled[y == 'normal']
autoencoder.fit(X_normal, X_normal, epochs=10, batch_size=64, validation_split=0.1, verbose=0)

# Reconstruction error
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 95)
y_pred_ae = np.where(mse > threshold, 'attack', 'normal')

print("\n===== Autoencoder (Deep Learning) Results =====")
print(classification_report(y, y_pred_ae))

# ===============================================================
# Step 9: Feature Importance (Random Forest)
# ===============================================================
print("\n📊 Plotting Feature Importance...")
rf_model = models_list["Random Forest"]
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title("Top 10 Important Features in NSL-KDD")
plt.show()

# ===============================================================
# Step 10: PCA Visualization (2D)
# ===============================================================
print("\n🎨 Generating PCA Visualization...")
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=df['label'], palette={'normal':'blue','attack':'red'})
plt.title("Network Traffic PCA Visualization (Normal vs Attack)")
plt.show()

# ===============================================================
# Step 11: ROC Curve (Random Forest)
# ===============================================================
print("\n📈 Plotting ROC Curve...")
y_prob = rf_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test.map({'normal':0, 'attack':1}), y_prob)
roc_auc = roc_auc_score(y_test.map({'normal':0, 'attack':1}), y_prob)

plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})', color='darkorange')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest (NSL-KDD)')
plt.legend()
plt.show()

# ===============================================================
# Step 12: Summary
# ===============================================================
print("\n✅ Final Model Comparison Table:")
print(results_df)
print("\n✅ Isolation Forest (Unsupervised) and Autoencoder (Deep Learning) also evaluated.")
print("\n🎯 Project Completed Successfully with NSL-KDD Dataset!")