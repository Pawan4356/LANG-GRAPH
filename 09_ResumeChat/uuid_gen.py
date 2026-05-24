import uuid

# UUID version 1 (based on host ID and current time)
print(uuid.uuid1())

# UUID version 3 (MD5 hash-based, requires namespace + name)
print(uuid.uuid3(uuid.NAMESPACE_DNS, "example.com"))

# UUID version 4 (random)
print(uuid.uuid4())

# UUID version 5 (SHA-1 hash-based, requires namespace + name)
print(uuid.uuid5(uuid.NAMESPACE_DNS, "example.com"))