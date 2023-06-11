"""
Simple script to
"""

import aes670hw2
root_token = "aes670hw2"

""" Assemble import strings for children in the provided namespace """
get_children = lambda token: set([
        f"{token}.{child}" # Assemble the child import strings
        for child in eval(f"dir({token})") # Get the namespace of the token
        if len(child)>4 and child[:2]+child[-2:]!="____" # Ignore dunders
        ])

"""  """
all_tokens = {}
children = get_children(root_token)
while len(children)>0:
    token = children.pop()
    all_tokens.add(token)
    children.add(children(token))

print(list(all_tokens))
