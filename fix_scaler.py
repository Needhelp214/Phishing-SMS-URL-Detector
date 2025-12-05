import pickle
import sklearn
from sklearn.preprocessing import StandardScaler

print(f"Current Scikit-learn version: {sklearn.__version__}")


with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Scaler updated successfully!")