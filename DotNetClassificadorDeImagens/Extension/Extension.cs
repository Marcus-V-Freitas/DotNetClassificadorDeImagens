using Microsoft.ML.Data;

namespace DotNetClassificadorDeImagens.Extension;

public static class Extension
{
    public static void ExibirMetricas(this MulticlassClassificationMetrics metricas)
    {
        Console.WriteLine($"Acurácia Macro = {metricas.MacroAccuracy:P2}");
        Console.WriteLine($"Acurácia Micro = {metricas.MicroAccuracy:P2}");
        Console.WriteLine(metricas.ConfusionMatrix.GetFormattedConfusionTable());
    }
}