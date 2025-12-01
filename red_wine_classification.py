import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import warnings

# Uyarıları kapat (daha temiz çıktı için)
warnings.filterwarnings('ignore')

# ==========================================
# 1. VERİ YÜKLEME VE ÖN İŞLEME
# ==========================================
print("1. VERİ YÜKLEME VE ÖN İŞLEME...")

# Veri setini yükle (ayırıcıyı otomatik kontrol etmeye çalışıyoruz, yaygın olarak ',' veya ';' kullanılır)
try:
    df = pd.read_csv('red_wine_cleaned.csv', sep=None, engine='python')
except Exception as e:
    print(f"Hata: Dosya okunamadı. {e}")
    exit()

print("Veri seti ilk 5 satır:")
print(df.head())

# Hedef değişkeni oluşturma: Quality >= 7 ise 1 (İyi), değilse 0 (Kötü)
df['quality_bin'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Eski 'quality' sütununu kaldırabiliriz veya analiz için tutabiliriz. Modele girmemesi lazım.
# X ve y ayırırken 'quality' ve 'quality_bin'i çıkaracağız.

# Hedef değişken dağılımını görselleştir
plt.figure(figsize=(6, 4))
sns.countplot(x='quality_bin', data=df)
plt.title('Hedef Değişken Dağılımı (0: Kötü, 1: İyi)')
plt.xlabel('Kalite Durumu')
plt.ylabel('Sayı')
plt.savefig('red_wine_dagilim.png')
plt.show()

print("\nHedef Değişken Dağılımı:")
print(df['quality_bin'].value_counts())

# ==========================================
# 2. KEŞİFSEL VERİ ANALİZİ (EDA)
# ==========================================
print("\n2. KEŞİFSEL VERİ ANALİZİ (EDA)...")

# Korelasyon matrisi
# Korelasyon matrisi
plt.figure(figsize=(12, 10))
# Sadece sayısal sütunları alalım (gerçi hepsi sayısal ama garanti olsun)
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Özellikler Korelasyon Matrisi')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('red_wine_korelasyon.png')
plt.show()

# ==========================================
# 3. VERİ BÖLME VE DENGESİZLİK GİDERME
# ==========================================
print("\n3. VERİ BÖLME VE DENGESİZLİK GİDERME...")

# Özellikler (X) ve Hedef (y) ayrımı
# 'quality' orijinal skor, 'quality_bin' bizim hedefimiz. İkisini de X'ten çıkarıyoruz.
# Ayrıca sayısal olmayan sütunları da çıkaralım (örn: 'type' sütunu varsa)
# 'target' sütunu da varsa (veri setinde önceden oluşturulmuş olabilir) onu da çıkaralım.
df_numeric = df.select_dtypes(include=[np.number])
drop_cols = ['quality', 'quality_bin']
if 'target' in df_numeric.columns:
    drop_cols.append('target')

X = df_numeric.drop(drop_cols, axis=1)
y = df['quality_bin']

# Eğitim ve Test seti ayrımı (%80 Train, %20 Test, Stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Eğitim seti boyutu (SMOTE öncesi): {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")

# SMOTE ile sınıf dengesizliğini giderme (Sadece Eğitim setine uygulanır!)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Eğitim seti boyutu (SMOTE sonrası): {X_train_smote.shape}")
print("SMOTE sonrası sınıf dağılımı:")
print(y_train_smote.value_counts())

# ==========================================
# 4. MODELLEME VE TUNING
# ==========================================
print("\n4. MODELLEME VE TUNING...")

# RandomForestClassifier tanımla
rf = RandomForestClassifier(random_state=42)

# Hiperparametre ızgarası (Grid)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, scoring='f1')
grid_search.fit(X_train_smote, y_train_smote)

best_model = grid_search.best_estimator_
print(f"\nEn İyi Parametreler: {grid_search.best_params_}")

# ==========================================
# 5. DEĞERLENDİRME VE KAYIT
# ==========================================
print("\n5. DEĞERLENDİRME VE KAYIT...")

# Test seti üzerinde tahmin
y_pred = best_model.predict(X_test)

# Metrikler
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.savefig('red_wine_confusion_matrix.png')
plt.show()

# Feature Importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.title("Özellik Önemi (Feature Importance)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45)
plt.tight_layout()
plt.savefig('red_wine_feature_importance.png')
plt.show()

# Modeli kaydet
model_filename = 'red_wine_model.pkl'
joblib.dump(best_model, model_filename)
print(f"\nEn iyi model '{model_filename}' adıyla kaydedildi.")
