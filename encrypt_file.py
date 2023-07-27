from cryptography.fernet import Fernet

# generate a new key
key = Fernet.generate_key()

# save the key to a file
with open("secret.key", "wb") as key_file:
    key_file.write(key)


# open the new file you want to encrypt
with open("plaintext.txt", "rb") as plaintext_file:
    plaintext = plaintext_file.read()


# initialize the fernet class using the key
cipher = Fernet(key)

# encrypt the plaintext
ciphertext = cipher.encrypt(plaintext)

# write the ciphertext to a file
with open("ciphertext.txt", "wb") as ciphertext_file:
    ciphertext_file.write(ciphertext)


