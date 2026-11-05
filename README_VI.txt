CHUONG TRINH DANH GIA SVD BEST RANK-k

1. CAU TRUC THU MUC DU LIEU

USC_SIPI_Dataset/
    A/       20 anh TIFF xam
    B/       20 anh TIFF xam
    C/       10 anh TIFF RGB

Co the dung ten Group_A, Group_B, Group_C thay cho A, B, C.
Mac dinh chuong trinh chi chap nhan anh 512x512 va KHONG resize anh.

2. CAI DAT

Mo Terminal trong thu muc chua chuong trinh va chay:
    pip install -r requirements.txt

3. CHAY CHUONG TRINH

    python run_svd_experiment.py --input "duong_dan/USC_SIPI_Dataset" --output "duong_dan/SVD_Results"

Vi du neu thu muc du lieu nam canh file Python:
    python run_svd_experiment.py

4. KET QUA

SVD_Results/
    results.csv              500 dong (50 anh x 10 nguong), neu du 50 anh hop le
    group_summary.csv        Trung binh va do lech chuan theo nhom, nguong
    validation/validation.csv Ket qua kiem tra TIFF, kich thuoc, so kenh
    reconstructions/         Anh TIFF khoi phuc tai 10 nguong tu 99% den 90%
    svd_factors/              He so U, s, Vt float32 trong NPZ khong nen
    comparisons/              Anh goc va 10 anh khoi phuc; moi anh co chu thich nguong nang luong va k
    spectra/                  Pho singular value va nang luong tich luy
    plots/                    Bieu do tong hop giua cac nhom
    errors.csv                Loi trong qua trinh phan ra / khoi phuc
    run_summary.txt           Tong ket va canh bao

5. CAC QUY UOC

- Anh xam: mot phep SVD tren mot ma tran.
- Anh RGB: tach R, G, B va phan ra doc lap; chon k rieng cho tung kenh.
- Truoc khi SVD, chuong trinh tru gia tri pixel trung binh rieng cua tung kenh:
      mu = mean(A)
      B = A - mu
  Sau do phan ra B = U Sigma V^T va tinh nang luong:
      eta(k) = sum(i=1..k, sigma_i^2) / sum(i=1..r, sigma_i^2)
  Anh khoi phuc: A_hat = mu + U_k Sigma_k V_k^T.
- Vi vay, phan tram nang luong trong results.csv la nang luong cua phan bien thien
  quanh muc sang trung binh (mean-centered energy), khong phai nang luong cua toan bo
  ma tran pixel goc. Cach nay tranh de muc sang nen chi phoi qua manh ket qua k.
- k la so thanh phan nho nhat de nang luong mean-centered tich luy dat nguong.
- Voi RGB, mu duoc tinh rieng cho tung kenh R, G, B va duoc luu cung he so SVD.
- MSE va PSNR tinh tren anh 8-bit sau khi lam tron va luu duoc.
- SSIM dung data_range=255; voi RGB, tinh tren ba kenh.
- Dung luong ly thuyet tinh voi he so float32, khong tinh header.
- Dung luong thuc te la kich thuoc NPZ khong nen chua cac he so float32.
- Ty le factor_archive/source_TIFF so sanh voi dung luong TIFF goc tren dia.
- Anh khong dung kich thuoc hoac sai loai kenh se bi loai, khong tu dong resize.

Luu y: anh TIFF RGB co the duoc nen san (LZW/Deflate); do do, ty le so voi TIFF
phu thuoc vao cach TIFF goc duoc ma hoa. Khi dien giai ket qua, can neu ro ca
raw_pixel_bytes, source_tiff_bytes va factor_archive_bytes.

6. CHU THICH k TREN ANH SO SANH

- Moi o anh khoi phuc trong comparisons/ hien thi nguong nang luong va k duoc chon.
- Anh xam hien thi k = ... .
- Anh RGB hien thi kR, kG, kB va tong Sigma k, vi moi kenh mau co mot k rieng.
- Ten tep TIFF trong reconstructions/ cung chua nguong nang luong va cac gia tri k.
- Khong ghi chu truc tiep len pixel cua anh TIFF khoi phuc, de khong lam thay doi anh
  dung de tinh MSE, PSNR va SSIM.
