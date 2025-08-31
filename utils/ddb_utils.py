"""
# Session State Management with DynamoDB

This module provides Pydantic models for session state serialization/deserialization and a DynamoDB helper class for managing chat sessions.

## Features

- **Flexible Serialization**: Automatically handles new properties and ignores absent properties
- **Type Safety**: Full type hints and Pydantic validation
- **DynamoDB Integration**: Complete CRUD operations for session management
- **Error Handling**: Comprehensive error handling for all database operations

## Models

### Message
Represents a single chat message in a session.

```python
class Message(BaseModel):
    role: str                    # "user" or "assistant"
    content: str                 # The message content
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)
```

### Session
Represents a complete chat session with messages and metadata.

```python
class Session(BaseModel):
    id: str                      # Unique session identifier
    messages: List[Message]      # List of chat messages
    metadata: Dict[str, Any]    # Flexible metadata storage
```

**Key Features:**
- `extra = "ignore"`: Ignores extra fields during deserialization
- `populate_by_name = True`: Allows flexible field population
- Automatic default values for optional fields

## DdbTable Helper Class

### Initialization
```python
ddb_table = DdbTable(table_name="chat-sessions", region_name="us-east-1")
```

### Methods

#### `create_session(session: Session) -> bool`
Creates a new session in DynamoDB.

```python
session = Session(id="123", messages=[], metadata={})
success = ddb_table.create_session(session)
```

#### `get_session(id: str) -> Optional[Session]`
Retrieves a session by ID.

```python
session = ddb_table.get_session("123")
if session:
    print(f"Found session with {len(session.messages)} messages")
```

#### `update_session(session: Session) -> bool`
Updates an existing session.

```python
session.messages.append(new_message)
session.metadata["updated"] = True
success = ddb_table.update_session(session)
```

#### `delete_session(id: str) -> bool`
Deletes a session by ID.

```python
success = ddb_table.delete_session("123")
```

#### `list_sessions(limit: int = 100) -> List[Session]`
Lists all sessions (up to the specified limit).

```python
sessions = ddb_table.list_sessions(limit=50)
for session in sessions:
    print(f"Session {session.id}: {len(session.messages)} messages")
```

## Usage Examples

### Basic Session Creation
```python
from utils.ddb_utils import Session, Message, DdbTable

# Create a session
session = Session(
    id="unique-session-id",
    messages=[
        Message(role="user", content="Hello!", additional_kwargs={}),
        Message(role="assistant", content="Hi there!", additional_kwargs={})
    ],
    metadata={"user_id": "user123", "created_at": "2024-01-01"}
)

# Save to DynamoDB
ddb_table = DdbTable("chat-sessions")
ddb_table.create_session(session)
```

### Flexible Deserialization
The models handle extra/missing fields gracefully:

```python
# Data with extra fields (will be ignored)
data = {
    "id": "123",
    "messages": [...],
    "extra_field": "ignored",  # This will be ignored
    "another_extra": {"nested": "data"}  # This too
}

# Missing fields will use defaults
minimal_data = {"id": "123"}  # messages=[], metadata={}

session = Session(**data)  # Works with both cases
```

### JSON Serialization
```python
# Serialize to JSON
json_str = session.model_dump_json(indent=2)

# Deserialize from JSON
session = Session.model_validate_json(json_str)
```

## Setup Requirements

1. **AWS Credentials**: Configure AWS credentials (via AWS CLI, environment variables, or IAM roles)
2. **DynamoDB Table**: Create a table with `id` as the primary key
3. **Dependencies**: Ensure `boto3` and `pydantic` are installed

### DynamoDB Table Schema
```json
{
  "TableName": "chat-sessions",
  "KeySchema": [
    {
      "AttributeName": "id",
      "KeyType": "HASH"
    }
  ],
  "AttributeDefinitions": [
    {
      "AttributeName": "id",
      "AttributeType": "S"
    }
  ]
}
```

## Error Handling

All methods include comprehensive error handling:

- **ClientError**: AWS/DynamoDB specific errors
- **General Exceptions**: Unexpected errors
- **Graceful Degradation**: Methods return `None` or `False` on failure

## Testing

Run the test script to verify functionality:

```bash
python utils/test_session_models.py
```

This will test:
- Session creation from example data
- JSON serialization/deserialization
- Flexible field handling
- Default value handling

## Example Data Structure

The models work with the exact structure from your example:

```json
{
  "id": "5b46e5cb-a770-441a-a85e-ed6bd714192d",
  "messages": [
    {
      "role": "user",
      "content": "You're an dApp AI assistant chatbot...",
      "additional_kwargs": {}
    },
    {
      "role": "assistant", 
      "content": "Since the question is related to code generation...",
      "additional_kwargs": {}
    }
  ],
  "metadata": {}
}
```

## Benefits

1. **Schema Evolution**: Add new fields to your session model without breaking existing data
2. **Type Safety**: Full IDE support and runtime validation
3. **Flexibility**: Handle various data formats and structures
4. **Reliability**: Comprehensive error handling and logging
5. **Performance**: Efficient DynamoDB operations with proper indexing 
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError
import json
from datetime import datetime
import uuid


class Message(BaseModel):
    """Model for chat messages in a session."""
    role: str
    content: str
    additional_kwargs: Dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """Model for session state with flexible serialization/deserialization."""
    id: str
    messages: List[Message] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        # Allow extra fields during deserialization (ignore absent properties)
        extra = "ignore"
        # Allow population by field name
        populate_by_name = True


class DdbTable:
    """Helper class for DynamoDB operations with session state."""
    
    def __init__(self, table_name: str, region_name: str = "us-east-1"):
        """
        Initialize DdbTable helper.
        
        Args:
            table_name: Name of the DynamoDB table
            region_name: AWS region name
        """
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
        self.table = self.dynamodb.Table(table_name)
    
    def create_session(self, session: Session) -> bool:
        """
        Create a new session in DynamoDB.
        
        Args:
            session: Session object to create
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if the session already exists
        if self.get_session(session.id):
            print(f"Session {session.id} already exists")
            return False
        
        try:
            # Convert session to dict for DynamoDB storage
            session_dict = session.model_dump()
            
            # Add timestamp for tracking
            session_dict['created_at'] = datetime.utcnow().isoformat()
            session_dict['updated_at'] = datetime.utcnow().isoformat()
            
            self.table.put_item(Item=session_dict)
            return True
        except ClientError as e:
            print(f"Error creating session: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error creating session: {e}")
            return False
    
    def get_session(self, id: str) -> Optional[Session]:
        """
        Retrieve a session by ID from DynamoDB.
        
        Args:
            id: Session ID to retrieve
            
        Returns:
            Session object if found, None otherwise
        """
        try:
            response = self.table.get_item(Key={'id': id})
            
            if 'Item' not in response:
                return None
            
            # Convert DynamoDB item to Session object
            # The flexible deserialization will handle any extra/missing fields
            session_data = response['Item']
            
            # Convert messages if they exist
            if 'messages' in session_data:
                messages = []
                for msg_data in session_data['messages']:
                    messages.append(Message(**msg_data))
                session_data['messages'] = messages
            
            return Session(**session_data)
            
        except ClientError as e:
            print(f"Error retrieving session: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error retrieving session: {e}")
            return None
    
    def delete_session(self, id: str) -> bool:
        """
        Delete a session by ID from DynamoDB.
        
        Args:
            id: Session ID to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.table.delete_item(Key={'id': id})
            return True
        except ClientError as e:
            print(f"Error deleting session: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error deleting session: {e}")
            return False
    
    def update_session(self, session: Session) -> bool:
        """
        Update an existing session in DynamoDB.
        
        Args:
            session: Session object to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert session to dict for DynamoDB storage
            session_dict = session.model_dump()
            
            # Update timestamp
            session_dict['updated_at'] = datetime.utcnow().isoformat()
            
            # Use update_item for atomic updates
            update_expression = "SET "
            expression_attribute_values = {}
            expression_attribute_names = {}
            
            # Build update expression dynamically
            update_parts = []
            for key, value in session_dict.items():
                if key != 'id':  # Skip the primary key
                    attr_name = f"#{key}"
                    attr_value = f":{key}"
                    update_parts.append(f"{attr_name} = {attr_value}")
                    expression_attribute_names[attr_name] = key
                    expression_attribute_values[attr_value] = value
            
            update_expression += ", ".join(update_parts)
            
            self.table.update_item(
                Key={'id': session.id},
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expression_attribute_names,
                ExpressionAttributeValues=expression_attribute_values
            )
            
            return True
        except ClientError as e:
            print(f"Error updating session: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error updating session: {e}")
            return False
    
    def list_sessions(self, limit: int = 100) -> List[Session]:
        """
        List all sessions (with optional limit).
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of Session objects
        """
        try:
            response = self.table.scan(Limit=limit)
            sessions = []
            
            for item in response.get('Items', []):
                # Convert messages if they exist
                if 'messages' in item:
                    messages = []
                    for msg_data in item['messages']:
                        messages.append(Message(**msg_data))
                    item['messages'] = messages
                
                sessions.append(Session(**item))
            
            return sessions
        except ClientError as e:
            print(f"Error listing sessions: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error listing sessions: {e}")
            return []


# Example usage and helper functions
def create_sample_session() -> Session:
    """Create a sample session for testing."""
    messages = [
        Message(
            role="user",
            content="Hello, I need help with Rust programming.",
            additional_kwargs={}
        ),
        Message(
            role="assistant", 
            content="I'd be happy to help you with Rust programming! What specific questions do you have?",
            additional_kwargs={}
        )
    ]
    
    return Session(
        id=str(uuid.uuid4()),
        messages=messages,
        metadata={"user_id": "test_user", "created_by": "system"}
    )


def serialize_session_to_json(session: Session) -> str:
    """Serialize session to JSON string."""
    return session.model_dump_json(indent=2)


def deserialize_session_from_json(json_str: str) -> Session:
    """Deserialize session from JSON string."""
    return Session.model_validate_json(json_str)


def example_usage():
    """Demonstrate how to use the DdbTable class."""
    
    # Initialize the DdbTable helper
    # Note: You'll need AWS credentials configured and a DynamoDB table created
    table_name = "chat-sessions"  # Replace with your actual table name
    region_name = "us-east-1"     # Replace with your AWS region
    
    try:
        ddb_table = DdbTable(table_name, region_name)
        print(f"✅ Connected to DynamoDB table: {table_name}")
    except Exception as e:
        print(f"⚠️  Could not connect to DynamoDB: {e}")
        print("This is expected if AWS credentials are not configured.")
        print("The example will show the interface without actual database operations.")
        return
    
    # Example 1: Create a new session
    print("\n📝 Example 1: Creating a new session")
    
    # Create a sample session
    session = create_sample_session()
    print(f"Created session with ID: {session.id}")
    
    # Save to DynamoDB
    success = ddb_table.create_session(session)
    if success:
        print("✅ Session saved to DynamoDB")
    else:
        print("❌ Failed to save session to DynamoDB")
    
    # Example 2: Retrieve a session
    print(f"\n📖 Example 2: Retrieving session {session.id}")
    
    retrieved_session = ddb_table.get_session(session.id)
    if retrieved_session:
        print(f"✅ Retrieved session: {retrieved_session.id}")
        print(f"   Messages: {len(retrieved_session.messages)}")
        print(f"   Metadata: {retrieved_session.metadata}")
    else:
        print("❌ Session not found")
    
    # Example 3: Update a session
    print(f"\n✏️  Example 3: Updating session {session.id}")
    
    # Add a new message to the session
    new_message = Message(
        role="user",
        content="Can you help me with more Rust examples?",
        additional_kwargs={}
    )
    session.messages.append(new_message)
    
    # Update metadata
    session.metadata["last_updated"] = "2024-01-01"
    session.metadata["message_count"] = len(session.messages)
    
    # Save the updated session
    success = ddb_table.update_session(session)
    if success:
        print("✅ Session updated in DynamoDB")
        print(f"   New message count: {len(session.messages)}")
        print(f"   Updated metadata: {session.metadata}")
    else:
        print("❌ Failed to update session")
    
    # Example 4: List all sessions
    print(f"\n📋 Example 4: Listing all sessions")
    
    sessions = ddb_table.list_sessions(limit=10)
    print(f"Found {len(sessions)} sessions:")
    for s in sessions:
        print(f"  - {s.id}: {len(s.messages)} messages")
    
    # Example 5: Delete a session
    print(f"\n🗑️  Example 5: Deleting session {session.id}")
    
    success = ddb_table.delete_session(session.id)
    if success:
        print("✅ Session deleted from DynamoDB")
    else:
        print("❌ Failed to delete session")
    
    # Verify deletion
    retrieved_session = ddb_table.get_session(session.id)
    if retrieved_session is None:
        print("✅ Confirmed: Session no longer exists")
    else:
        print("❌ Session still exists after deletion")


def example_with_custom_data():
    """Example using the provided session data structure."""
    
    print("\n🔧 Example with custom session data:")
    
    # Create a session with the exact structure from your example
    custom_session = Session(
        id="5b46e5cb-a770-441a-a85e-ed6bd714192d",
        messages=[
            Message(
                role="user",
                content="You're an dApp AI assistant chatbot...",
                additional_kwargs={}
            ),
            Message(
                role="assistant",
                content="Since the question is related to code generation...",
                additional_kwargs={}
            )
        ],
        metadata={}
    )
    
    print(f"✅ Created custom session: {custom_session.id}")
    print(f"   Messages: {len(custom_session.messages)}")
    print(f"   First message role: {custom_session.messages[0].role}")
    print(f"   Second message role: {custom_session.messages[1].role}")
    
    # Serialize to JSON
    import json
    session_json = custom_session.model_dump_json(indent=2)
    print(f"\n📄 Serialized JSON (first 200 chars):")
    print(session_json[:200] + "...")
    
    # Deserialize from JSON
    deserialized = Session.model_validate_json(session_json)
    print(f"\n✅ Deserialized session ID: {deserialized.id}")


def test_with_example_data():
    """Test the models with the provided example data."""
    
    # Example data from the user
    example_data = {
        "id": "5b46e5cb-a770-441a-a85e-ed6bd714192d",
        "messages": [
            {
                "role": "user",
                "content": "\nYou're an dApp AI assistant chatbot, which helps the user to answer questions related to a coding co-pilot, which is used for code completion tasks in Rust language, \nspecifically: Substrate and Ink frameworks.\n\nIMPORTANT:\n1. If the question is related to code completion or code generation, then use the context data that are provided. \n2. If the question is about code documentation, features of the co-pilot or something else, then don't use the provided context data.\n\nInformation:\n\nThe dApp AI Co-pilot is a VSCode plugin for Substrate and ink! development. You can help answering the questions to the user about:\n\n1. Explain how to install and set up the dApp AI Co-pilot VSCode plugin, including the requirement for GitHub Token authentication.\n\n2. Provide information on best practices for using the co-pilot effectively with Substrate and ink! frameworks.\n\n3. Answer questions about code explanation that you have generated.\n\n4. Help the user with code refactoring.\n\n5. Help the user for writing test for specific code.\n\n6. Offer guidance on how users can leverage the co-pilot for learning and improving their Substrate and ink! coding skills.\n\nAnswer the query below:\nwrite a simple macro that creates storagemaps\n\nSelected Code:\nwarp_sync\n\nWhen answering your questions keep in mind the conversation history too. \n",
                "additional_kwargs": {}
            },
            {
                "role": "assistant",
                "content": "Since the question is related to code generation for Substrate/Ink!, I will use the provided context to assist you.\n\nHere's a simple macro that creates storagemaps in Ink!:\n\n```rust\n#[macro_export]\nmacro_rules! create_storagemaps {\n    ($vis:vis $name:ident<$key:ty, $value:ty>: $pun:ty) => {\n        $vis struct $name;\n\n        #[ink(storage)]\n        $vis mod $name {\n            use ink_storage::collections::HashMap as StorageMap;\n\n            pub struct $name {\n                map: StorageMap<$key, $value>,\n            }\n\n            impl $name {\n                #[ink(constructor)]\n                pub fn new() -> Self {\n                    Self {\n                        map: StorageMap::new(),\n                    }\n                }\n\n                #[ink(message)]\n                pub fn insert(&mut self, key: $key, value: $value) {\n                    self.map.insert(key, value);\n                }\n\n                #[ink(message)]\n                pub fn get(&self, key: &$key) -> Option<$value> {\n                    self.map.get(key).cloned()\n                }\n            }\n        }\n    };\n}\n```\n\nThis macro defines a new struct and module with the given name. The module contains a `StorageMap` that maps keys of type `$key` to values of type `$value`. The macro also generates a `new` constructor and `insert` and `get` messages for interacting with the storage map.\n\nYou can use this macro like:\n\n```rust\ncreate_storagemaps!(pub AccountBalances<AccountId, Balance>: ink_storage::traits::StorageLayout);\n```\n\nThis will create a `pub` module `AccountBalances` with a storage map that maps `AccountId` keys to `Balance` values.\n\nLet me know if you need any clarification or have additional requirements!",
                "additional_kwargs": {}
            }
        ],
        "metadata": {}
    }
    
    print("Testing session model with example data...")
    
    # Test 1: Create session from example data
    try:
        session = Session(**example_data)
        print("✅ Successfully created session from example data")
        print(f"Session ID: {session.id}")
        print(f"Number of messages: {len(session.messages)}")
        print(f"Metadata keys: {list(session.metadata.keys())}")
    except Exception as e:
        print(f"❌ Error creating session: {e}")
        return
    
    # Test 2: Serialize to JSON
    try:
        json_str = serialize_session_to_json(session)
        print("\n✅ Successfully serialized session to JSON")
        print(f"JSON length: {len(json_str)} characters")
    except Exception as e:
        print(f"❌ Error serializing session: {e}")
        return
    
    # Test 3: Deserialize from JSON
    try:
        deserialized_session = deserialize_session_from_json(json_str)
        print("✅ Successfully deserialized session from JSON")
        print(f"Deserialized session ID: {deserialized_session.id}")
        print(f"Deserialized messages count: {len(deserialized_session.messages)}")
    except Exception as e:
        print(f"❌ Error deserializing session: {e}")
        return
    
    # Test 4: Test flexible deserialization with extra fields
    try:
        # Add some extra fields that shouldn't be in the model
        extra_data = example_data.copy()
        extra_data["extra_field"] = "this should be ignored"
        extra_data["another_extra"] = {"nested": "data"}
        
        flexible_session = Session(**extra_data)
        print("✅ Successfully handled extra fields (ignored them)")
        print(f"Session still has correct ID: {flexible_session.id}")
        
        # Verify extra fields are not in the model
        session_dict = flexible_session.model_dump()
        assert "extra_field" not in session_dict
        assert "another_extra" not in session_dict
        print("✅ Extra fields were properly ignored")
        
    except Exception as e:
        print(f"❌ Error testing flexible deserialization: {e}")
        return
    
    # Test 5: Test missing fields (should use defaults)
    try:
        minimal_data = {"id": "test-id"}
        minimal_session = Session(**minimal_data)
        print("✅ Successfully created session with minimal data")
        print(f"Default messages: {len(minimal_session.messages)}")
        print(f"Default metadata: {minimal_session.metadata}")
        
    except Exception as e:
        print(f"❌ Error testing minimal data: {e}")
        return
    
    print("\n🎉 All tests passed! The session models work correctly.")


def test_ddb_table_interface():
    """Test the DdbTable interface (without actual DynamoDB calls)."""
    
    print("\nTesting DdbTable interface...")
    
    # Note: This would require actual AWS credentials and DynamoDB table
    # For now, just show the interface
    try:
        # This would fail without proper AWS setup, but shows the interface
        ddb_table = DdbTable("test-sessions-table")
        print("✅ DdbTable interface is properly defined")
        print("Available methods:")
        print("  - create_session(session: Session) -> bool")
        print("  - get_session(id: str) -> Optional[Session]")
        print("  - delete_session(id: str) -> bool")
        print("  - update_session(session: Session) -> bool")
        print("  - list_sessions(limit: int = 100) -> List[Session]")
        
    except Exception as e:
        print(f"ℹ️  DdbTable requires AWS setup: {e}")
        print("The interface is ready to use with proper AWS credentials")


if __name__ == "__main__":
    print("🚀 DdbTable Helper Class Tests")
    print("=" * 50)
    test_with_example_data()
    test_ddb_table_interface() 
    print("\n")

    print("🚀 DdbTable Helper Class Examples")
    print("=" * 50)
    
    example_usage()
    example_with_custom_data()
    
    print("\n" + "=" * 50)
    print("✨ Examples completed!")
    print("\nTo use this in your application:")
    print("1. Configure AWS credentials")
    print("2. Create a DynamoDB table with 'id' as the primary key")
    print("3. Update the table_name and region_name in the code")
    print("4. Import and use the DdbTable class in your application") 