import firebase_admin
from firebase_admin import credentials, firestore, storage

# Initialize Firebase Admin SDK
cred = credentials.Certificate('serviceAccountKey.json')  # Download this from Firebase console
firebase_admin.initialize_app(cred, {
    'storageBucket': 'notesapp-dixitk941.appspot.com'  # Replace with your project ID
})

db = firestore.client()
bucket = storage.bucket()