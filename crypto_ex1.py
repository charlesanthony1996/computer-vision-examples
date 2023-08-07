from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC 
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
import os
import base64

# this is the input from the user
password_provided = "password"
password = password_provided.encode()
salt = os.urandom(16)
kdf = PBKDF2HMAC(
    algorithm = hashes.SHA256(),
    length = 32,
    salt = salt,
    iterations = 100000,
)

key = base64.urlsafe_b64encode(kdf.derive(password))

print(key)


# save the key to a file
with open("secret.key", "wb") as key_file:
    key_file.write(key)


# load the key
with open("secret.key", "rb") as key_file:
    key = key_file.read()


# open the file you want to encrypt 
try:
    with open("plaintext.txt", "rb") as plaintext_file:
        plaintext = plaintext_file.read()

except FileNotFoundError:
    print("the file to be encrpted was not found")
    exit(1)

# initialize the fernet class using the new key
cipher = Fernet(key)

# print(cipher)

# encrypt the plaintext
ciphertext = cipher.encrypt(plaintext)


# enter the ciphertext to a file
with open("ciphertext.txt", "wb") as ciphertext_file:
    ciphertext_file.write(ciphertext)


# to decrypt the file
cipher = Fernet(key)
try:
    with open("ciphertext.txt", "rb") as enc_file:
        encrypted_data = enc_file.read()
except FileNotFoundError:
    print('the file to be decrypted was not found')
    exit(1)


try:
    decrypted_data = cipher.decrypt(encrypted_data)
except InvalidToken:
    print("The key used for decryption is incorrect")
    exit(1)


with open("decrypted_plaintext.txt", "wb") as dec_file:
    dec_file.write(decrypted_data)



