from roboflow import Roboflow
from ultralytics import YOLO

def main():
    # ==========================================
    # 1. تحميل مجموعة البيانات (Dataset) من Roboflow
    # ==========================================
    # الرجاء إدخال تفاصيل المشروع الخاص بك أو أي مشروع من Roboflow Universe
    # مفتاحك الذي استخدمته سابقاً هو: q4iQCtkpyfbQuwfSA3eK
    
    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key="q4iQCtkpyfbQuwfSA3eK")
    
    # استبدل هذه ببيانات مشروعك الفعلي على Roboflow 
    # (يمكنك الحصول على هذا الكود من زر Export Dataset في الموقع)
    project = rf.workspace("kreems-workspace").project("drone-or-bird-classifier")
    version = project.version(1) # تأكد من رقم النسخة لديك
    dataset = version.download("yolov8")
    
    data_yaml_path = f"{dataset.location}/data.yaml"
    print(f"Dataset downloaded at: {data_yaml_path}")
    
    # ==========================================
    # 2. إعداد الموديل للتدريب
    # ==========================================
    print("Loading base model...")
    model = YOLO('yolov8n.pt')  # الموديل الأساسي الذي سنبني عليه
    
    # ==========================================
    # 3. بدء عملية التدريب
    # ==========================================
    print("Starting training process...")
    # يمكنك تعديل القيم أدناه حسب قوة جهازك
    results = model.train(
        data=data_yaml_path,
        epochs=50,       # عدد دورات التدريب (كلما زاد الرقم زادت الدقة، 50 ممتاز كبداية)
        imgsz=640,       # دقة الصور (حجم قياسي)
        device='0',      # استخدام كرت الشاشة (إذا واجهت مشكلة اجعلها 'cpu')
        batch=16         # عدد الصور التي تتم معالجتها معاً (اجعلها 8 إذا امتلأت ذاكرة الكرت)
    )
    
    print("==========================================")
    print("Training finished!")
    print("You can find the best model weights at: runs/detect/train/weights/best.pt")
    print("==========================================")

if __name__ == '__main__':
    main()
