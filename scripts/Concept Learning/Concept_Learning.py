import sys
train_file=sys.argv[1]
file = open(train_file,'r')
list_of_attributes=[]
target_value=[]
for line in file:
    attributes=[]
    for k in range(0,len(line)-2,2):
        attributes.append(line[k])
    target_value.append(line[k+2])
    list_of_attributes.append(attributes)
file.close()
# Initialize with most specific hypothesis
literals_space=[]
for k in range(len(list_of_attributes)):
    if target_value[k]=='1':
        for i in range(len(attributes)):
            if list_of_attributes[k][i]=='1':
                literal=i+1
            else:
                literal=-i-1
            literals_space.append(literal)
    if len(literals_space)>=1:
        train_example_number=k
        break
    else:
        pass
for j in range(train_example_number,len(list_of_attributes)):
    if target_value[j]=='1':
        for m in range(len(attributes)):
            if list_of_attributes[j][m]=='1':
                literal_check=m+1
            else:
                literal_check=-m-1
            if (literal_check*(-1)) in literals_space:
                literals_space.remove(literal_check*(-1))
print(len(literals_space), *literals_space, sep = ", ")