import time

# sample message data
message_list = [
    {"definition": {"id":"error", "priority": 4 }},
    {"definition": {"id": "warning", "priority": 3}},
    {"definition": {"id": "info", "priority": 1}},
    {"definition": {"id": "warning", "priority": 2}},
]

# print(message_list)

def sort_by_priority(messages):
    return sorted(messages, key=lambda x: x['definition']['priority'])


# function to filter messages by id prefix and then sort by priority
def get_messages_by_id_starting_with(partial_id):
    # filter messages based on whether their id starts with provided substring
    filtered_messages = [entry for entry in message_list if entry['definition']['id'].startswith(partial_id)]
    # sort the filtered messages by priority and return
    return sort_by_priority(filtered_messages)


# test the function
filtered_and_sorted = get_messages_by_id_starting_with("warning")
# print(filtered_and_sorted)

for message in filtered_and_sorted:
    print(message['definition']['id'], "-", message["definition"]["priority"])