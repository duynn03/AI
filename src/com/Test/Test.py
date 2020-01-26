def example1():
    for i in range(3):
        x = int(input("enter a number: "))
        y = int(input("enter another number: "))
        try:
            print(x, '/', y, '=', x / y)
        except ZeroDivisionError:
            print("Can't divide by 0!")
        except ValueError:
            print("That doesn't look like a number!")
        except:
            print("something unexpected happend!")


example1()