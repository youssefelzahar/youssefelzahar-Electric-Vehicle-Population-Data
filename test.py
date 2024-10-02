import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# مثال على بيانات
data = {'Column1': [10, 12, 11, 13, 100, 15], 'Column2': [50, 52, 51, 53, 200, 55]}
df = pd.DataFrame(data)

# بناء نموذج Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)

# تدريب النموذج بناءً على العمودين معًا
df['outlier'] = model.fit_predict(df[['Column1', 'Column2']])

# عرض النتائج
outliers = df[df['outlier'] == -1]
print("القيم المتطرفة بناءً على علاقة العمودين:")
print(outliers)

# عرض البيانات باستخدام scatter plot
plt.scatter(df['Column1'], df['Column2'], color='blue', label='Normal Points')
plt.scatter(outliers['Column1'], outliers['Column2'], color='red', label='Outliers')
plt.xlabel('Column1')
plt.ylabel('Column2')
plt.legend()
plt.show()
