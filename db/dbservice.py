import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError
from dotenv import load_dotenv

# Load environment variables from the .env file in the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path)

class DBService:
    """
    MongoDB service for storing Zenity ROV telemetry logs.
    Follows OOP principles for connection and data insertion.
    """
    
    def __init__(self, db_name="rov_db", collection_name="sessions"):
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        
        # Initialize connection on creation
        self.connect()

    def connect(self):
        """Establishes a connection to MongoDB using the URI from environment variables."""
        mongo_url = os.getenv("MONGODB_URL")
        
        if not mongo_url:
            print("[DBService] ⚠ Warning: MONGODB_URL not found in .env file. Database logging will be disabled.")
            return

        try:
            # Set a short timeout so it doesn't block forever if the URI is wrong or network is down
            self.client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
            # Force a connection test
            self.client.admin.command('ping')
            
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            print(f"[DBService] ✓ Successfully connected to MongoDB Database: {self.db_name}")
            
        except (ConnectionFailure, ConfigurationError) as e:
            print(f"[DBService] ✗ Failed to connect to MongoDB: {e}")
            self.client = None

    def save_session(self, session_data: dict) -> bool:
        """
        Saves a complete session document to MongoDB.
        
        Args:
            session_data (dict): A dictionary containing session metadata and the list of logs.
            
        Returns:
            bool: True if insertion was successful, False otherwise.
        """
        if self.client is None or self.collection is None:
            print("[DBService] ⚠ Cannot save session: MongoDB connection is not established.")
            return False
        
        try:
            print(f"[DBService] Saving session '{session_data.get('session_id')}' with {session_data.get('total_logs', 0)} logs...")
            result = self.collection.insert_one(session_data)
            print(f"[DBService] ✓ Session saved successfully. Document ID: {result.inserted_id}")
            return True
        except Exception as e:
            print(f"[DBService] ✗ Error saving session to MongoDB: {e}")
            return False

    def close(self):
        """Closes the MongoDB connection."""
        if self.client:
            self.client.close()
            print("[DBService] MongoDB connection closed.")
