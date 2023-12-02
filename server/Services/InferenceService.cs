using Microsoft.Extensions.ML;
using Microsoft.ML;
using Microsoft.ML.Data;
using Server.Models;

namespace Server.Services;

public interface IInferenceService
{
    Dictionary<string, float> Predict(byte[] image, string filename);
}

public class InferenceService(MLContext mlContext, PredictionEnginePool<ModelInput, ModelOutput> predictionEnginePool) : IInferenceService
{
    private readonly MLContext _mlContext = mlContext;
    private readonly PredictionEnginePool<ModelInput, ModelOutput> _predictionEnginePool = predictionEnginePool;

    public  Dictionary<string, float> Predict(byte[] image, string filename)
    {
        ModelInput modelInput = new()
        {
            Image = image,
            ImagePath = filename
        };

        ModelOutput modelOutput = _predictionEnginePool.Predict(modelInput);
        
        var scores = GetScoresWithLabelsSorted(_predictionEnginePool.GetPredictionEngine().OutputSchema,"Score", modelOutput.Probability);

        return scores;
    }

    private static Dictionary<string, float> GetScoresWithLabelsSorted(DataViewSchema schema, string name, float[] scores)
        {
            Dictionary<string, float> result = [];

            var column = schema.GetColumnOrNull(name);

            var slotNames = new VBuffer<ReadOnlyMemory<char>>();
            column!.Value.GetSlotNames(ref slotNames);
            var names = new string[slotNames.Length];
            var num = 0;
            foreach (var denseValue in slotNames.DenseValues())
            {
                result.Add(denseValue.ToString(), scores[num++]);
            }

            return result.OrderByDescending(c => c.Value).ToDictionary(i => i.Key, i => i.Value);
        }
}