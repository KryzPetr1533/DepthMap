#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <string>
#include <iostream>

// Параметры для настройки камеры глубины
int numDisparities = 48; // Устанавливает диапазон значений несоответствия для поиска
int blockSize = 5; // Размер скользящего окна, используемого для сопоставления блоков, чтобы найти соответствующие пиксели в выпрямленной паре стереоизображений. Более высокое значение указывает на больший размер окна
int preFilterType = 1; // Параметр, определяющий тип предварительной фильтрации, применяемой к изображениям перед передачей в алгоритм сопоставления блоков. Этот шаг расширяет информацию о текстуре и улучшает результаты алгоритма сопоставления блоков. Тип фильтра может быть CV_STEREO_BM_XSOBEL или CV_STEREO_BM_NORMALIZED_RESPONSE
int preFilterSize = 5; // Размер окна фильтра, используемого на этапе предварительной фильтрации
int preFilterCap = 31; // Ограничивает отфильтрованный вывод до определенного значения
int minDisparity = 5; // Минимальное значение несоответствия для поиска
int textureThreshold = 10; // Отфильтровывает области, в которых недостаточно информации о текстуре для надежного сопоставления
int uniquenessRatio = 15; //  Еще один этап постфильтрации. Пиксель отфильтровывается, если наилучшее несоответствие недостаточно лучше, чем любое другое несоответствие в диапазоне поиска. Следующий GIF показывает, что увеличение коэффициента уникальности увеличивает количество отфильтрованных пикселей
int speckleRange = 0; // Определяет, насколько близкими должны быть значения несоответствия, чтобы их можно было рассматривать как часть одного и того же блоба
int speckleWindowSize = 0; // Число пикселей, ниже которого капля несоответствия отклоняется как «крапинка»
int disp12MaxDiff = -1; // определяет максимально допустимую разницу между исходным левым пикселем и пикселем с обратным соответствием.
int dispType = CV_16S;


int main(){
    /*
    // Шахматная доска
    int CHECKERBOARD[2]{ 9, 9 };
    // Расположение папки с фотографиями для калибровки
    std::string path = "C:\\Users\\guyju\\source\\repos\\DepthMap\\DepthMap\\CameraCalibTest\\";
    // Расширение тестовых файлов для калибровки
    std::string extention = ".jpg";
    // Создание массива векторов точек доски в реальности
    std::vector<std::vector<cv::Point3f> > objpoints;
    // Создание массива вектров 2d точек для каждой камеры
    std::vector<std::vector<cv::Point2f> > imgpointsL, imgpointsR;
    // Создание и заполнение координат системы в реальности
    std::vector<cv::Point3f> objp;
    for(int i{0}; i<CHECKERBOARD[1]; i++){
        for(int j{0}; j<CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(j,i,0));
    }
    // Количество изображений в тестовой папке
    int n{ 7 };
    // Название изображений
    std::string imageL = "imgL", imageR = "imgR";
    // Изображение камер
    cv::Mat frameL, frameR, grayL, grayR;
    // Массив для хранения координат углов доски
    std::vector<cv::Point2f> corner_ptsL, corner_ptsR;
    // Проверка успеха
    bool successL, successR;
    // Looping over all the images in the directory
    for (int i{ 0 }; i < n; i++){
        std::string nameL = path + imageL + std::to_string(i + 1) + extention;
        frameL = cv::imread(nameL);
        cv::cvtColor(frameL, grayL, cv::COLOR_BGR2GRAY);
        std::string nameR = path + imageR + std::to_string(i + 1) + extention;
        frameR = cv::imread(nameR);
        cv::cvtColor(frameR, grayR, cv::COLOR_BGR2GRAY);
        // Проверка нахождения углов 
        successL = cv::findChessboardCorners(grayL, cv::Size(CHECKERBOARD[0],CHECKERBOARD[1]), 
            corner_ptsL, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        successR = cv::findChessboardCorners(grayR,cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), 
            corner_ptsR, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        if((successL) && (successR)){
            cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);
            // Проверка координат пикселей для точек
            cv::cornerSubPix(grayL,corner_ptsL,cv::Size(11,11), cv::Size(-1,-1),criteria);
            cv::cornerSubPix(grayR,corner_ptsR,cv::Size(11,11), cv::Size(-1,-1),criteria);
            cv::drawChessboardCorners(frameL, cv::Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_ptsL,successL);
            cv::drawChessboardCorners(frameR, cv::Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_ptsR,successR);
            // Добавляем данные
            objpoints.push_back(objp);
            imgpointsL.push_back(corner_ptsL);
            imgpointsR.push_back(corner_ptsR);
        }
        cv::imshow("ImageL", frameL);
        cv::imshow("ImageR", frameR);
        cv::waitKey(0);
    }
    cv::destroyAllWindows();
    // Stereo калибровка
    cv::Mat mtxL, distL, R_L, T_L;
    cv::Mat mtxR, distR, R_R, T_R;
    cv::Mat new_mtxL, new_mtxR;
    // Калибровка левой камеры
    cv::calibrateCamera(objpoints, imgpointsL, grayL.size(), mtxL, distL, R_L, T_L);
    new_mtxL = cv::getOptimalNewCameraMatrix(mtxL, distL, grayL.size(), 1, grayL.size(), 0);
    // Калибровка правой камеры
    cv::calibrateCamera(objpoints, imgpointsR, grayR.size(), mtxR, distR, R_R, T_R);
    new_mtxR = cv::getOptimalNewCameraMatrix(mtxR, distR, grayR.size(), 1, grayR.size(), 0);
    // Параметры стерео системы
    cv::Mat Rot, Trns, Emat, Fmat;
    int flag = 0;
    flag |= cv::CALIB_FIX_INTRINSIC;
    // Калибровка стерео системы
    cv::stereoCalibrate(objpoints, imgpointsL, imgpointsR, new_mtxL, distL, new_mtxR, distR, grayR.size(), Rot, Trns, Emat, Fmat, flag, cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 1e-6));
    cv::Mat rect_l, rect_r, proj_mat_l, proj_mat_r, Q;
    // Выравнивание плоскостей изображений в одну плоскость
    cv::stereoRectify(new_mtxL, distL, new_mtxR, distR, grayR.size(), Rot, Trns, rect_l, rect_r, proj_mat_l, proj_mat_r, Q, 1);
    */
    cv::Mat Left_Stereo_Map1, Left_Stereo_Map2;
    cv::Mat Right_Stereo_Map1, Right_Stereo_Map2;
    // Находим матрицу выравнивания
    //cv::initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l, grayR.size(), CV_16SC2, Left_Stereo_Map1, Left_Stereo_Map2);
    //cv::initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r, grayR.size(), CV_16SC2, Right_Stereo_Map1, Right_Stereo_Map2);
    
    std::string name = "C:\\Users\\guyju\\source\\repos\\DepthMap\\DepthMap\\StereoSystemCharacteristic.xml";
    //cv::FileStorage cv_file = cv::FileStorage(name, cv::FileStorage::WRITE);
    //cv_file.write("Left_Stereo_Map_x",Left_Stereo_Map1);
    //cv_file.write("Left_Stereo_Map_y",Left_Stereo_Map2);
    //cv_file.write("Right_Stereo_Map_x",Right_Stereo_Map1);
    //cv_file.write("Right_Stereo_Map_y",Right_Stereo_Map2);
    //cv_file.release();
    
   // Объект класса StereoBM
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(numDisparities, blockSize);
    cv::Mat imgL, imgR, imgL_gray, imgR_gray;
    cv::Mat disp, disparity, disp_8;
    stereo->setPreFilterType(preFilterType);
    stereo->setPreFilterSize(preFilterSize);
    stereo->setPreFilterCap(preFilterCap);
    stereo->setTextureThreshold(textureThreshold);
    stereo->setUniquenessRatio(uniquenessRatio);
    stereo->setSpeckleRange(speckleRange);
    stereo->setSpeckleWindowSize(speckleWindowSize);
    stereo->setDisp12MaxDiff(disp12MaxDiff);
    stereo->setMinDisparity(minDisparity);
    cv::FileStorage cv_file2 = cv::FileStorage(name, cv::FileStorage::READ);
    cv_file2["Left_Stereo_Map_x"] >> Left_Stereo_Map1;
    cv_file2["Left_Stereo_Map_y"] >> Left_Stereo_Map2;
    cv_file2["Right_Stereo_Map_x"] >> Right_Stereo_Map1;
    cv_file2["Right_Stereo_Map_y"] >> Right_Stereo_Map2;
    cv_file2.release();
    int CamL_id{ 0 }, CamR_id{ 1 };

    cv::VideoCapture camL(CamL_id), camR(CamR_id);
    camL.set(cv::CAP_PROP_BUFFERSIZE, 3);
    camR.set(cv::CAP_PROP_BUFFERSIZE, 3);
    if (!camL.isOpened()){
        std::cout << "Could not open camera with index : " << CamL_id << std::endl;
        return -1;
    }
    if (!camR.isOpened()){
        std::cout << "Could not open camera with index : " << CamR_id << std::endl;
        return -1;
    }
    cv::namedWindow("disparity", cv::WINDOW_NORMAL);
    //cv::resizeWindow("disparity", 800, 800);
    float sixteen = 16.0;
    float minDisparityf = 5.0;
    float numDisparitiesf = 48.0;
    while (true){
        camL.read(imgL);
        camR.read(imgR);
        cv::cvtColor(imgL, imgL_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgR, imgR_gray, cv::COLOR_BGR2GRAY);
        // Матрицы для выровненных изображений 
        cv::Mat Left_nice, Right_nice;
        // Выравнивание
        cv::remap(imgL_gray, Left_nice, Left_Stereo_Map1, Left_Stereo_Map2, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
        cv::remap(imgR_gray, Right_nice, Right_Stereo_Map1, Right_Stereo_Map2, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, 0);
        // Вычисление диспа несоответствия
        stereo->compute(Left_nice, Right_nice, disp);
        // Перевод из CV_16S в CV_32F
        disp.convertTo(disparity, CV_32F, 1.0);
        // Нормализавация
        
        disparity = (disparity / sixteen - minDisparityf) / (numDisparitiesf);
        // Карта несоответствия disparity map
        cv::imshow("disparity", disparity);
        // Esc
        if (cv::waitKey(1) == 27) {
            break;
        }
    }
    //*/
    return 0;
}
