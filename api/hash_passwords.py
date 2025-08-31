import bcrypt

def hash_password(plain_password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(plain_password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')

if __name__ == "__main__":
    passwords = ["Merxh!en4Lifn"]
    for password in passwords:
        print(f"Plain: {password}, Hashed: {hash_password(password)}")
