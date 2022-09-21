import json
dict=[]

# with open('points_robo.txt', 'r') as file:
#     info=file.read().rstrip()
#     with open("js.json",'w') as js:
#         json.dump(info,js)

    #i=info.replace("'",'"')
    # e=info.replace("'",'"')
    # info = json.loads(e)
    # print(info)





    # print(info)
    # #info='{}'

    # print(type(info))
    # print(info)

with open("js.json",'r') as jso:
    points=json.load(jso)
    print(points[0])
    print(type(points))
    print(len(points))
    #print(points)



