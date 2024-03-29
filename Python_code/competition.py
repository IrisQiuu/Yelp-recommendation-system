'''
Method Description:
I developed a hybrid recommendation system that incorporates the predictions from item-based collaborative filtering as a feature within an XGBoost model used for predicting ratings. To enhance the xgboost model's performance, I enriched it with additional features extracted from the business.json dataset, such as location coordinates (latitude and longitude), price range, and business categories. As a result of these adjustments, there are no predictions falling within the error category of >=4, leading to a notable reduction in RMSE. While this marks significant progress, I intend to continue exploring alternative methods to further refine the recommendation system. As of now, this stands as the most effective recommendation system I've developed.

Error Distribution:
>=0 and <1:  103610
>=1 and <2:  32048
>=2 and <3:  5708
>=3 and <4:  678
>=4:  0

RMSE:
0.95699

Execution Time:
310s

RMSE(Test dataset):
0.97624

'''
import sys
import math
import time
import os
from pyspark import SparkConf, SparkContext
import json
import numpy as np
import xgboost as xgb #!pip3 install xgboost==0.72.1
from sklearn.metrics import mean_squared_error

#train_folder = '/Users/chenyuqiu/DSCI553/hw3/task2';test_file = '/Users/chenyuqiu/DSCI553/hw3/task2/yelp_val.csv';output_file = '/Users/chenyuqiu/DSCI553/hw3/task2/out_comp.csv'

train_folder = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

def get_data(data_train, data_test):
    train_feature = []
    train_rating = []
    test_feature = []
    for item in data_train:
        feature = []
        if user_map.get(item[0]):
            feature.extend(user_map.get(item[0]))
            if business_map.get(item[1]):
                feature.extend(business_map.get(item[1]))
            else:
                feature.extend([avg_business_rating, avg_business_reviewcnt, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            feature.extend([avg_users_reviewcnt, avg_users_rating, 0, 0, 0, 0, 0, 0])
            if business_map.get(item[1]):
                feature.extend(business_map.get(item[1]))
            else:
                feature.extend([avg_business_rating, avg_business_reviewcnt, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0])

        train_feature.append(feature)
        train_rating.append(item[2])
    train_feature = np.asarray(train_feature)
    train_rating = np.asarray(train_rating)

    for item_test in data_test:
        feature_test = []
        if user_map.get(item_test[0]):
            feature_test.extend(user_map.get(item_test[0]))
            if business_map.get(item_test[1]):
                feature_test.extend(business_map.get(item_test[1]))
            else:
                feature_test.extend([avg_business_rating, avg_business_reviewcnt, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            feature_test.extend([avg_users_reviewcnt, avg_users_rating, 0, 0, 0, 0, 0, 0])
            if business_map.get(item_test[1]):
                feature_test.extend(business_map.get(item_test[1]))
            else:
                feature_test.extend([avg_business_rating, avg_business_reviewcnt, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0])

        test_feature.append(feature_test)
    test_feature = np.asarray(test_feature)

    return train_feature, train_rating, test_feature


def get_price_range(attributes, key):
    if attributes:
        if key in attributes.keys():
            return int(attributes.get(key))
    return 0

def getAtt(attributes, key, mapping):
    if attributes and key in attributes:
        value = attributes[key]
        if value in mapping:
            return mapping[value]
        else:
            return 0
    return 0

def getCategories(catlist):
    category_mapping = {
        "Restaurants": 0,
        "Food": 1,
        "Shopping": 2,
        "Home Services": 3,
        "Beauty & Spas": 4,
        "Nightlife": 5,
        "Health & Medical": 6,
        "Local Services": 7,
        "Bars": 8,
        "Automotive": 9
    }
    return category_mapping.get(next((cat for cat in catlist if cat in category_mapping), 10))


def getCorr(bid_r, uid_rate, bid_avg):
    neighbour_avg = bid_meanRate[bid_r]
    uid_r = bid_set_uid_rate[bid_r]
    business_rate_list = []
    neighbour_rate_list = []
    for this_uid in list(uid_rate):  
        if uid_r[this_uid]:
            business_rate_list.append(uid_rate[this_uid])
            neighbour_rate_list.append(uid_r[this_uid])
    if not business_rate_list:
        numerator = 0
        denominator_business = 0
        denominator_neighbour = 0
        for i in range(len(business_rate_list)):
            normalized_bid_r = business_rate_list[i] - bid_avg
            normalized_neighbour_r = neighbour_rate_list[i] - neighbour_avg
            numerator += normalized_bid_r * normalized_neighbour_r
            denominator_business += normalized_bid_r ** 2
            denominator_neighbour += normalized_neighbour_r ** 2
        denominator = math.sqrt(denominator_business * denominator_neighbour)
        if numerator == 0 or denominator == 0:
            corr_value = 0
        else:
            corr_value = numerator / denominator
    else:
        corr_value = bid_avg / neighbour_avg
    return corr_value


def getPred(corr_list):
    w_sum = 0
    corr_sum = 0
    num_neighbour = min(30, len(corr_list))
    sorted_corr = sorted(corr_list, key=lambda x: x[0], reverse=True)
    for i in range(num_neighbour):
        w_sum += sorted_corr[i][0] * sorted_corr[i][0]
        corr_sum += abs(sorted_corr[i][0])
    pred_rate = w_sum / corr_sum
    return min(5.0, pred_rate)


def itemBasedCF(uid_bid):
    uid = uid_bid[0]  
    bid = uid_bid[1]  
    if bid not in bid_set_uid_rate:
        if not uid_set_bid_rate[uid]:
            return [uid, bid, '4.0']
        return [uid, bid, str(uid_meanRate[uid])]
    else:
        uid_rate = bid_set_uid_rate[bid]
        bid_avg = bid_meanRate[bid]
        if not uid_set_bid_rate[uid]:
            return [uid, bid, str(bid_meanRate[bid])]
        else:
            bid_rated = list(uid_set_bid_rate[uid])
            if not bid_rated:
                corr_list = []
                for bid_r in bid_rated:  
                    get_rate = bid_set_uid_rate[bid_r][uid]
                    corr = getCorr(bid_r, uid_rate, bid_avg)
                    if corr > 0.3:
                        corr_list.append((corr, get_rate))
                if not corr_list:
                    pred = min(5.0, getPred(corr_list))
                else:
                    pred = min(5.0, (uid_meanRate[uid] + bid_avg) / 2)
                return [uid, bid, str(pred)]
            else:
                return [uid, bid, str(bid_avg)]

def countFriends(friends_list):
    num_friends = len(friends_list) if friends_list[0] != "None" else 0
    return num_friends

def getCity(city, city_mapping):
    if city in city_mapping:
        return city_mapping[city]
    else:
        return 0

sc = SparkContext("local[*]", "competition").getOrCreate()
sc.setLogLevel("ERROR")
start = time.time()
data_train = sc.textFile(os.path.join(train_folder, 'yelp_train.csv'))
data_val = sc.textFile(os.path.join(train_folder, 'yelp_val.csv'))
data_train = data_train.union(data_val)
header_train = data_train.first()
data_train = data_train.filter(lambda x: x != header_train).map(lambda x: x.split(",")).cache()#.map(lambda x: (x[0], x[1], float(x[2]))).collect()

bid_set_uid_rate = data_train.map(lambda r: (r[1], (r[0], float(r[2])))).groupByKey().map(lambda x: (x[0], dict(x[1]))).sortByKey().collectAsMap()
uid_set_bid_rate = data_train.map(lambda r: (r[0], (r[1], float(r[2])))).groupByKey().map(lambda x: (x[0], dict(x[1]))).sortByKey().collectAsMap()
bid_meanRate = data_train.map(lambda r: (r[1], float(r[2]))).groupByKey().mapValues(lambda values: sum(values)/len(values)).collectAsMap()
uid_meanRate = data_train.map(lambda r: (r[0], float(r[2]))).groupByKey().mapValues(lambda values: sum(values)/len(values)).collectAsMap()

data_test = sc.textFile(test_file)
header_test = data_test.first()
data_test = data_test.filter(lambda r: r != header_test).map(lambda r: r.split(",")).cache()#print(data_test.take(2))
itembaseRDD_train = data_train.map(itemBasedCF).map(lambda x: (((x[0]), (x[1])), float(x[2])))
data_train_modelbase = data_train.map(lambda x: ((x[0], x[1]), float(x[2]))).join(itembaseRDD_train).map(lambda x: (x[0][0], x[0][1], x[1][0], x[1][1])).collect()
itembaseRDD = data_test.map(itemBasedCF).map(lambda x: (((x[0]), (x[1])), float(x[2])))
data_test_modelbase = itembaseRDD.map(lambda x: (x[0][0], x[0][1], x[1])).collect()


#load user and business data
user_rdd = sc.textFile(os.path.join(train_folder, 'user.json')) \
    .map(json.loads) \
    .map(lambda x: ((x["user_id"]), (x["review_count"], x["average_stars"], x["useful"], x["fans"], countFriends(x.get("friends").split(", ")), x['compliment_more'], x['compliment_plain'], x['cool']))).cache()#x['compliment_writer'],

user_map = user_rdd.collectAsMap()
avg_users_reviewcnt = user_rdd.map(lambda x: x[1][0]).mean()
avg_users_rating = user_rdd.map(lambda x: x[1][1]).mean()
wifi_mapping = {"no": 1, "paid": 2, "free": 3}
noise_mapping = {"quiet": 1, "average": 2, "loud": 3, "very_loud": 4}
tf_mapping = {"false": 1, "true": 2}
city_mapping = {"Las Vegas": 1, "Phoenix": 2, "Toronto": 3, "Charlotte": 4, "Scottsdale": 5, "Pittsburgh": 6, "Mesa": 7, "Montréal": 8, "Henderson": 9, "Tempe": 10}

business_rdd = sc.textFile(os.path.join(train_folder, 'business.json')) \
    .map(json.loads) \
    .map(lambda x: ((x['business_id']), (x['stars'], x['review_count'], x['latitude'], x['longitude'],
    get_price_range(x['attributes'], 'RestaurantsPriceRange2'), getCategories(
    x.get("categories").split(', ') if x.get('categories') else []), x['is_open'],
    getAtt(x['attributes'], 'WiFi', wifi_mapping), getAtt(x['attributes'], 'NoiseLevel', noise_mapping),
    getAtt(x['attributes'], 'HasTV', tf_mapping), #getAtt(x['attributes'], 'GoodForKids', tf_mapping),
    getAtt(x['attributes'], 'RestaurantsDelivery', tf_mapping), getAtt(x['attributes'], 'RestaurantsReservations', tf_mapping),
    getCity(x['city'], city_mapping)))).cache()

business_map = business_rdd.collectAsMap()
avg_business_rating = business_rdd.map(lambda x: x[1][0]).mean()
avg_business_reviewcnt = business_rdd.map(lambda x: x[1][1]).mean()
train_x, train_y, test_x = get_data(data_train_modelbase, data_test_modelbase)


xgbmodel = xgb.XGBRegressor(max_depth=7, learning_rate=0.1, verbosity=0, n_estimators=220, random_state=2)
xgbmodel.fit(train_x, train_y)
test_pred = xgbmodel.predict(test_x)

with open(output_file, 'w') as file:
    file.write("user_id, business_id, prediction\n")
    for i in range(0, len(test_pred)):
        file.write(data_test_modelbase[i][0] + "," + data_test_modelbase[i][1] + "," + str(
            min(5, max(1, test_pred[i]))) + "\n")

end = time.time()
print("Execution Time:", end - start)

'''RMSE= 0.9792575504313681

outputRDD = sc.textFile(output_file)
outputFirstRow = outputRDD.first()
outputRDD = outputRDD.filter(lambda x: x != outputFirstRow).map(lambda x: x.split(',')).map(lambda x: (((x[0]), (x[1])), float(x[2])))
data_test = sc.textFile(test_file)
header_test = data_test.first()
data_test = data_test.filter(lambda x: x != header_test).map(lambda x: x.split(",")).map(lambda x: (((x[0]), (x[1])), float(x[2])))
error_RDD = data_test.join(outputRDD).map(lambda x: (abs(x[1][0] - x[1][1])))
print(">=0 and <1: ", error_RDD.filter(lambda x: x >= 0 and x < 1).count())
print(">=1 and <2: ", error_RDD.filter(lambda x: x >= 1 and x < 2).count())
print(">=2 and <3: ", error_RDD.filter(lambda x: x >= 2 and x < 3).count())
print(">=3 and <4: ", error_RDD.filter(lambda x: x >= 3 and x < 4).count())
print(">=4: ", error_RDD.filter(lambda x: x >= 4).count())


rmseRDD = error_RDD.map(lambda x: x ** 2).reduce(lambda x, y: x + y)
print("RMSE=", math.sqrt(rmseRDD / outputRDD.count()))
'''
sc.stop()