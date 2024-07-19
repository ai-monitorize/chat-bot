from retriever import retrieve

def main():

    while True:
        prompt = input("prompt: ")

        if prompt == "quit":
            break
        else:
            print(retrieve(prompt))

main()