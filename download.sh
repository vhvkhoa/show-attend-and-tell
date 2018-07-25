TRAINYEAR=2017
VALYEAR=2017
TESTYEAR=2014

if [! $TRAINYEAR -eq $VALYEAR]
then
    wget http://images.cocodataset.org/annotations/annotations_trainval$TRAINYEAR.zip -P data/
    unzip annotations_trainval$TRAINYEAR.zip
    rm annotations_trainval$TRAINYEAR.zip
fi

wget http://images.cocodataset.org/annotations/annotations_trainval$VALYEAR.zip -P data/
wget http://images.cocodataset.org/annotations/image_info_test$TESTYEAR.zip -P data/

wget http://images.cocodataset.org/zips/train$TRAINYEAR.zip -P image/
wget http://images.cocodataset.org/zips/val$VALYEAR.zip -P image/
wget http://images.cocodataset.org/zips/test$TESTYEAR.zip -P image/

unzip data/annotations_trainval$VALYEAR.zip -d data/

unzip data/image_info_test$TESTYEAR.zip -d data/
unzip image/train$TRAINYEAR.zip -d image/ && mv image/train$TRAINYEAR image/train
unzip image/val$VALYEAR.zip -d image/ && mv image/val$VALYEAR image/val
unzip image/test$TESTYEAR.zip -d image/ && mv image/test$TESTYEAR image/test

rm image/train$TRAINYEAR.zip
rm image/val$VALYEAR.zip
rm image/test$TESTYEAR.zip
rm data/annotations_trainval$VALYEAR.zip
rm data/image_info_test$TESTYEAR.zip

mkdir data/train
mkdir data/val
mkdir data/test
