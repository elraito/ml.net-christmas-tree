using System.Text.Json;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;
using Server.Models;
using static Microsoft.ML.DataOperationsCatalog;

namespace Server.Services;

public interface IModelBuilderService
{
    void TrainModel();
}

public class ModelBuilderService(MLContext mlContext) : IModelBuilderService
{
    private readonly MLContext _mlContext = mlContext;

    public void TrainModel()
    {
        string projectDirectory = "/data";
        string workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
        string assetsRelativePath = Path.Combine(projectDirectory, "dataset");

        IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);

        IDataView imageData = _mlContext.Data.LoadFromEnumerable(images);
        IDataView shuffledData = _mlContext.Data.ShuffleRows(imageData);

        var preprocessingPipeline = _mlContext.Transforms.Conversion.MapValueToKey(
                inputColumnName: nameof(ModelInput.Label),
                outputColumnName: nameof(ModelInput.LabelAsKey))
            .Append(_mlContext.Transforms.LoadRawImageBytes(
                outputColumnName: nameof(ModelInput.Image),
                imageFolder: assetsRelativePath,
                inputColumnName: nameof(ModelInput.ImagePath)
            ));

        IDataView preProcessedData = preprocessingPipeline
            .Fit(shuffledData)
            .Transform(shuffledData);

        TrainTestData trainSplit = _mlContext.Data.TrainTestSplit(preProcessedData, testFraction: 0.3);
        TrainTestData validationSplit = _mlContext.Data.TrainTestSplit(trainSplit.TestSet);

        IDataView trainSet = trainSplit.TrainSet;
        IDataView validationSet = validationSplit.TrainSet;
        IDataView testSet = validationSplit.TestSet;

        ImageClassificationTrainer.Options classifierOptions = new()
        {
            FeatureColumnName = nameof(ModelInput.Image),
            LabelColumnName = nameof(ModelInput.LabelAsKey),
            ValidationSet = validationSet,
            Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
            MetricsCallback = Console.WriteLine,
            TestOnTrainSet = false,
            ReuseTrainSetBottleneckCachedValues = true,
            ReuseValidationSetBottleneckCachedValues = true
        };

        EstimatorChain<KeyToValueMappingTransformer> trainingPipeline =
            _mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
            .Append(_mlContext.Transforms.Conversion.MapKeyToValue(nameof(ModelOutput.PredictedLabel)));

        ITransformer trainedModel = trainingPipeline.Fit(trainSet);

        _mlContext.Model.Save(trainedModel, trainSet.Schema, Path.Combine(workspaceRelativePath, "model.zip"));
    }

    static void ClassifySingleImage(MLContext mLContext, IDataView data, ITransformer trainedModel)
    {
        PredictionEngine<ModelInput, ModelOutput> predictionEngine = mLContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
        ModelInput image = mLContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).First();
        ModelOutput prediction = predictionEngine.Predict(image);
        Console.WriteLine("Classification:");        
        Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} | Value: {prediction.PredictedLabel}");
        Console.WriteLine($"Probabilities: {JsonSerializer.Serialize(prediction)}");
    }

    private static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
    {
        var files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);

        foreach(var file in files)
        {
            if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                continue;

            var label = Path.GetFileName(file);

            if (useFolderNameAsLabel)
                label = Directory.GetParent(file)!.Name;
            else 
            {
                for (int index = 0; index < label.Length; index++)
                {
                    if (!char.IsLetter(label[index]))
                    {
                        label = label[..index];
                        break;
                    }
                }
            }

            yield return new ImageData()
            {
                ImagePath = file,
                Label = label
            };
        }
    }
}