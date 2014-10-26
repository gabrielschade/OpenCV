#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
 

using namespace cv;

Mat escala_cinza(Mat imagem, bool mostrar);
Mat criarKernelSharpness();

int calcularResultadoPixel(Mat imagem, int x, int y, Mat kernel);
int calculaPosicao(Mat imagem, int x, int y, int x2, int y2, Mat kernel);
void negativa(Mat imagem);
void limiarizacao(Mat imagem, int limiar);
void somar(Mat imagem1, Mat imagem2);
void subtrair(Mat imagem1, Mat imagem2);
void mostrarImagem(char* janela, Mat imagem);
void aplicarRoberts(Mat imagem);
void aplicarSobel(Mat imagem);
void aplicarRobinson(Mat imagem);
void aplicarLaplace(Mat imagem);
void aplicarCanny(Mat imagem);

double modifiedLaplacian(Mat src);
double varianceOfLaplacian(Mat src);
double tenengrad(Mat src, int ksize);
double normalizedGraylevelVariance(Mat src);

Mat criarKernelRoberts(int tipo);
Mat criarKernelRobinson(int tipo, bool negativo = false);


int main(int argc, char** argv)
{
	/*char* nome_imagem = "casa.jpg";
	char* nome_imagem2 = "soma.jpg";*/

	char* nome_imagem = "Imagens/16000.tif";

	/*char* nome_imagem2 = "janela1.png";*/

	int limiar = 100;

	Mat imagemOriginal = imread(nome_imagem);
	Mat imagem;
	GaussianBlur(imagemOriginal, imagem, Size(3, 3), 0, 0, BORDER_DEFAULT);

	double x = modifiedLaplacian(imagemOriginal);
	double y = varianceOfLaplacian(imagemOriginal);
	double z = tenengrad(imagemOriginal, 3);
	double v = normalizedGraylevelVariance(imagemOriginal);

	printf("%d", x);
	printf("\n");
	printf("%d", y);
	printf("\n");
	printf("%d", z);
	printf("\n");
	printf("%d", v);
	printf("\n");
	/*limiarizacao(imagem, 100);
	aplicarRobinson(imagem);
	aplicarRoberts(imagem);
	aplicarSobel(imagem);
	aplicarLaplace(imagem);
	aplicarCanny(imagem);*/
	waitKey(0);
}

#pragma region Operações com Imagens

void somar(Mat imagem1, Mat imagem2)
{
	Mat imagemResultado;
	if (imagem1.size().area() > imagem2.size().area())
		imagem1.copyTo(imagemResultado);
	else
		imagem2.copyTo(imagemResultado);

	for (int i = 0; i < imagemResultado.rows - 1; i++)
		for (int j = 0; j < imagemResultado.cols - 1; j++)
		{
		if (imagem2.rows > i && imagem2.cols > j)
		{
			imagemResultado.at<Vec3b>(i, j)[0] = (imagem1.at<Vec3b>(i, j)[0] + imagem2.at<Vec3b>(i, j)[0]) / 2;
			imagemResultado.at<Vec3b>(i, j)[1] = (imagem1.at<Vec3b>(i, j)[1] + imagem2.at<Vec3b>(i, j)[1]) / 2;
			imagemResultado.at<Vec3b>(i, j)[2] = (imagem1.at<Vec3b>(i, j)[2] + imagem2.at<Vec3b>(i, j)[2]) / 2;
		}
		else
		{
			imagemResultado.at<Vec3b>(i, j)[0] = (imagem1.at<Vec3b>(i, j)[0]);
			imagemResultado.at<Vec3b>(i, j)[1] = (imagem1.at<Vec3b>(i, j)[1]);
			imagemResultado.at<Vec3b>(i, j)[2] = (imagem1.at<Vec3b>(i, j)[2]);
		}
		}

	mostrarImagem("soma", imagemResultado);
}
void subtrair(Mat imagem1, Mat imagem2)
{
	Mat imagemResultado;
	if (imagem1.size().area() > imagem2.size().area())
		imagem1.copyTo(imagemResultado);
	else
		imagem2.copyTo(imagemResultado);

	for (int i = 0; i < imagemResultado.rows - 1; i++)
		for (int j = 0; j < imagemResultado.cols - 1; j++)
		{
		if (imagem2.rows > i && imagem2.cols > j)
		{
			imagemResultado.at<Vec3b>(i, j)[0] = (imagem1.at<Vec3b>(i, j)[0] - imagem2.at<Vec3b>(i, j)[0]);
			imagemResultado.at<Vec3b>(i, j)[1] = (imagem1.at<Vec3b>(i, j)[1] - imagem2.at<Vec3b>(i, j)[1]);
			imagemResultado.at<Vec3b>(i, j)[2] = (imagem1.at<Vec3b>(i, j)[2] - imagem2.at<Vec3b>(i, j)[2]);

			int diferencaR = imagemResultado.at<Vec3b>(i, j)[0] - imagem1.at<Vec3b>(i, j)[0];
			int diferencaG = imagemResultado.at<Vec3b>(i, j)[1] - imagem1.at<Vec3b>(i, j)[1];
			int diferencaB = imagemResultado.at<Vec3b>(i, j)[2] - imagem1.at<Vec3b>(i, j)[2];

			if (
				(!(imagemResultado.at<Vec3b>(i, j)[0] == 0 &&
				imagemResultado.at<Vec3b>(i, j)[1] == 0 &&
				imagemResultado.at<Vec3b>(i, j)[2] == 0))
				)
			{
				imagemResultado.at<Vec3b>(i, j)[0] = (imagem1.at<Vec3b>(i, j)[0]);
				imagemResultado.at<Vec3b>(i, j)[1] = (imagem1.at<Vec3b>(i, j)[1]);
				imagemResultado.at<Vec3b>(i, j)[2] = (imagem1.at<Vec3b>(i, j)[2]);
			}
		}
		else
		{
			imagemResultado.at<Vec3b>(i, j)[0] = (imagem1.at<Vec3b>(i, j)[0]);
			imagemResultado.at<Vec3b>(i, j)[1] = (imagem1.at<Vec3b>(i, j)[1]);
			imagemResultado.at<Vec3b>(i, j)[2] = (imagem1.at<Vec3b>(i, j)[2]);
		}
		}


	mostrarImagem("subtrair", imagemResultado);
}
void negativa(Mat imagem)
{
	Mat negativa;
	imagem.copyTo(negativa);

	for (int i = 0; i < negativa.rows - 1; i++)
		for (int j = 0; j < negativa.cols - 1; j++)
		{
		negativa.at<Vec3b>(i, j)[0] = 255 - negativa.at<Vec3b>(i, j)[0];
		negativa.at<Vec3b>(i, j)[1] = 255 - negativa.at<Vec3b>(i, j)[1];
		negativa.at<Vec3b>(i, j)[2] = 255 - negativa.at<Vec3b>(i, j)[2];
		}


	mostrarImagem("Negativa", negativa);
}
void limiarizacao(Mat imagem, int limiar)
{
	Mat imagem_limiar;
	Mat imagem_cinza = escala_cinza(imagem, false);

	cv::threshold(imagem, imagem_limiar, limiar, 255,THRESH_BINARY_INV);
	/*for (int i = 0; i < imagem_cinza.rows - 1; i++)
		for (int j = 0; j < imagem_cinza.cols - 1; j++)
		{
		if (imagem_limiar.at<Vec3b>(i, j)[0] > limiar)
		{
			imagem_limiar.at<Vec3b>(i, j)[0] = 255;
			imagem_limiar.at<Vec3b>(i, j)[1] = 255;
			imagem_limiar.at<Vec3b>(i, j)[2] = 255;
		}
		else
		{
			imagem_limiar.at<Vec3b>(i, j)[0] = 0;
			imagem_limiar.at<Vec3b>(i, j)[1] = 0;
			imagem_limiar.at<Vec3b>(i, j)[2] = 0;
		}



		}*/

	mostrarImagem("Limiarizacao", imagem_limiar);
}
void aplicarRoberts(Mat imagem)
{
	Mat input = escala_cinza(imagem, false);
	Mat resultado, resultadoX, resultadoY;
	Mat kernelX = criarKernelRoberts(0);
	Mat kernelY = criarKernelRoberts(1);

	cv::filter2D(input, resultadoX, input.depth(), kernelX);
	cv::filter2D(input, resultadoY, input.depth(), kernelY);

	resultado = cv::abs(resultadoX) + cv::abs(resultadoY);
	mostrarImagem("Roberts", resultado);
}
void aplicarSobel(Mat imagem)
{
	Mat input = escala_cinza(imagem, false);
	Mat resultadoX, resultadoY, resultado;
	cv::Sobel(input, resultadoX, CV_8U, 1, 0);
	cv::Sobel(input, resultadoY, CV_8U, 0, 1);

	resultado = abs(resultadoX) + abs(resultadoY);
	mostrarImagem("Sobel", resultado);
}
void aplicarRobinson(Mat imagem)
{
	Mat input = escala_cinza(imagem, false);
	Mat resultadoX, resultadoXN, resultadoY, resultadoYN, resultadoDP, resultadoDPN, resultadoDS, resultadoDSN;
	Mat resultado;
	int depth = input.depth();

	cv::filter2D(input, resultadoX, depth, criarKernelRobinson(1));
	cv::filter2D(input, resultadoXN, depth, criarKernelRobinson(1, true));
	cv::filter2D(input, resultadoDP, depth, criarKernelRobinson(2));
	cv::filter2D(input, resultadoDPN, depth, criarKernelRobinson(2, true));
	cv::filter2D(input, resultadoY, depth, criarKernelRobinson(3));
	cv::filter2D(input, resultadoYN, depth, criarKernelRobinson(3, true));
	cv::filter2D(input, resultadoDS, depth, criarKernelRobinson(4));
	cv::filter2D(input, resultadoDSN, depth, criarKernelRobinson(4, true));

	max(resultadoX, abs(resultadoXN), resultado);
	max(resultado, resultadoY, resultado);
	max(resultado, resultadoYN, resultado);
	max(resultado, resultadoDP, resultado);
	max(resultado, resultadoDPN, resultado);
	max(resultado, resultadoDS, resultado);
	max(resultado, resultadoDSN, resultado);

	mostrarImagem("Robinson", resultado);
}
void aplicarLaplace(Mat imagem)
{
	Mat escala_cinza, resultado, abs_resultado;
	cvtColor(imagem, escala_cinza, CV_RGB2GRAY);

	Laplacian(escala_cinza, resultado, CV_16S, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(resultado, abs_resultado);
	mostrarImagem("Laplace", abs_resultado);
}
void aplicarCanny(Mat imagem)
{
	Mat resultado, escala_cinza;
	cvtColor(imagem, escala_cinza, CV_RGB2GRAY);
	/*cv::threshold(escala_cinza, resultado, 160, 255, THRESH_BINARY_INV);*/
	cv::Canny(imagem, resultado, 10, 350);
	mostrarImagem("Canny", resultado);
}

#pragma endregion

#pragma region Operadores de Foco

//LAPM
double modifiedLaplacian(cv::Mat src)
{
	cv::Mat M = (Mat_<double>(3, 1) << -1, 2, -1);
	cv::Mat G = cv::getGaussianKernel(3, -1, CV_64F);

	cv::Mat Lx;
	cv::sepFilter2D(src, Lx, CV_64F, M, G);

	cv::Mat Ly;
	cv::sepFilter2D(src, Ly, CV_64F, G, M);

	cv::Mat FM = cv::abs(Lx) + cv::abs(Ly);

	double focusMeasure = cv::mean(FM).val[0];
	return focusMeasure;
}

//LAPV
double varianceOfLaplacian(cv::Mat src)
{
	cv::Mat lap;
	cv::Laplacian(src, lap, CV_64F);

	cv::Scalar mu, sigma;
	cv::meanStdDev(lap, mu, sigma);

	double focusMeasure = sigma.val[0] * sigma.val[0];
	return focusMeasure;
}

//TENG
double tenengrad(Mat src, int ksize)
{
	cv::Mat Gx, Gy;
	cv::Sobel(src, Gx, CV_64F, 1, 0, ksize);
	cv::Sobel(src, Gy, CV_64F, 0, 1, ksize);

	cv::Mat FM = Gx.mul(Gx) + Gy.mul(Gy);

	double focusMeasure = cv::mean(FM).val[0];
	return focusMeasure;
}

//GLVN
double normalizedGraylevelVariance(Mat src)
{
	cv::Scalar mu, sigma;
	cv::meanStdDev(src, mu, sigma);

	double focusMeasure = (sigma.val[0] * sigma.val[0]) / mu.val[0];
	return focusMeasure;
}

#pragma endregion




Mat escala_cinza(Mat imagem, bool mostrar)
{
	Mat escala_cinza;
	cv::cvtColor(imagem, escala_cinza, CV_BGR2GRAY);

	if (mostrar)
		mostrarImagem("Escala Cinza", escala_cinza);

	return escala_cinza;
}
void mostrarImagem(char* janela, Mat imagem)
{
	namedWindow(janela);
	imshow(janela, imagem);
}

///tipo = 0 - kernel X, tipo !=0 - kernel Y
Mat criarKernelRoberts(int tipo)
{
	Mat kernel(2, 2, CV_32F);

	if (tipo == 0)
	{
		kernel.at<float>(0, 0) = 0;
		kernel.at<float>(0, 1) = 1;
		kernel.at<float>(1, 0) = -1;
		kernel.at<float>(1, 1) = 0;
	}
	else
	{
		kernel.at<float>(0, 0) = 1;
		kernel.at<float>(0, 1) = 0;
		kernel.at<float>(1, 0) = 0;
		kernel.at<float>(1, 1) = -1;
	}

	return kernel;
}

///Tipo = 1 - kernel X, Tipo = 2 - kernel DP, Tipo 3 - kernel Y, Tipo 4 - kernel DS
Mat criarKernelRobinson(int tipo, bool negativo)
{
	Mat kernel(3, 3, CV_32F);

	if (tipo == 1)
	{
		kernel.at<float>(0, 0) = 1;
		kernel.at<float>(0, 1) = 0;
		kernel.at<float>(0, 2) = -1;
		kernel.at<float>(1, 0) = 2;
		kernel.at<float>(1, 1) = 0;
		kernel.at<float>(1, 2) = -2;
		kernel.at<float>(2, 0) = 1;
		kernel.at<float>(2, 1) = 0;
		kernel.at<float>(2, 2) = -1;
	}
	else if (tipo == 2)
	{
		kernel.at<float>(0, 0) = 2;
		kernel.at<float>(0, 1) = 1;
		kernel.at<float>(0, 2) = 0;
		kernel.at<float>(1, 0) = 1;
		kernel.at<float>(1, 1) = 0;
		kernel.at<float>(1, 2) = -1;
		kernel.at<float>(2, 0) = 0;
		kernel.at<float>(2, 1) = -1;
		kernel.at<float>(2, 2) = -2;
	}
	else if (tipo == 3)
	{
		kernel.at<float>(0, 0) = 1;
		kernel.at<float>(0, 1) = 2;
		kernel.at<float>(0, 2) = 1;
		kernel.at<float>(1, 0) = 0;
		kernel.at<float>(1, 1) = 0;
		kernel.at<float>(1, 2) = 0;
		kernel.at<float>(2, 0) = -1;
		kernel.at<float>(2, 1) = -2;
		kernel.at<float>(2, 2) = -1;
	}
	else if (tipo == 4)
	{
		kernel.at<float>(0, 0) = 0;
		kernel.at<float>(0, 1) = -1;
		kernel.at<float>(0, 2) = -2;
		kernel.at<float>(1, 0) = 1;
		kernel.at<float>(1, 1) = 0;
		kernel.at<float>(1, 2) = -1;
		kernel.at<float>(2, 0) = 2;
		kernel.at<float>(2, 1) = 1;
		kernel.at<float>(2, 2) = 0;
	}

	if (negativo)
	{
		for (int i = 0; i < kernel.rows; i++)
			for (int j = 0; j < kernel.cols; j++)
				kernel.at<float>(i, j) *= -1;

	}

	return kernel;
}

#ifdef _EiC
main(1, "drawing.c");
#endif