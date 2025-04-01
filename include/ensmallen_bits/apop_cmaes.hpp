#ifndef APOP_CMAES_HPP
#define APOP_CMAES_HPP

#include <ensmallen.hpp>

namespace ens {

template<typename SelectionPolicyType = FullSelection,
         typename TransformationPolicyType = EmptyTransformation<>,
         bool UseBIPOPFlag = true>
class aPOP_CMAES : public ActiveCMAES<SelectionPolicyType, TransformationPolicyType>
{
public:
    aPOP_CMAES(const size_t lambda = 0,
               const TransformationPolicyType& transformationPolicy =
               TransformationPolicyType(),
               const size_t batchSize = 32,
               const size_t maxIterations = 1000,
               const double tolerance = 1e-5,
               const SelectionPolicyType& selectionPolicy = SelectionPolicyType(),
               double stepSize = 0,
               const size_t maxRestarts = 9,
               const double populationFactor = 2,
               const size_t maxFunctionEvaluations = 1e9);

    /**
   * Set POP-CMA-ES specific parameters.
   */
    template<typename SeparableFunctionType,
             typename MatType,
             typename... CallbackTypes>
    typename MatType::elem_type Optimize(SeparableFunctionType& function,
                                         MatType& iterate,
                                         CallbackTypes&&... callbacks);

    //! Get the population factor.
    double PopulationFactor() const { return populationFactor; }
    //! Modify the population factor.
    double& PopulationFactor() { return populationFactor; }

    //! Get the maximum number of restarts.
    size_t MaxRestarts() const { return maxRestarts; }
    //! Modify the maximum number of restarts.
    size_t& MaxRestarts() { return maxRestarts; }

    //! Get the maximum number of function evaluations.
    size_t MaxFunctionEvaluations() const { return maxFunctionEvaluations; }
    //! Modify the maximum number of function evaluations.
    size_t& MaxFunctionEvaluations() { return maxFunctionEvaluations; }

    //! Get the BIPOP mode flag.
    static constexpr bool UseBIPOP() { return UseBIPOPFlag; }

private:
    //! Population factor
    double populationFactor;

    //! Maximum number of restarts.
    size_t maxRestarts;

    //! Maximum number of function evaluations.
    size_t maxFunctionEvaluations;

};

template<typename SelectionPolicyType, typename TransformationPolicyType, bool UseBIPOPFlag>
aPOP_CMAES<SelectionPolicyType,
           TransformationPolicyType,
           UseBIPOPFlag>::aPOP_CMAES(
    const size_t lambda,
    const TransformationPolicyType& transformationPolicy,
    const size_t batchSize,
    const size_t maxIterations,
    const double tolerance,
    const SelectionPolicyType& selectionPolicy,
    double stepSize,
    const size_t maxRestarts,
    const double populationFactor,
    const size_t maxFunctionEvaluations) :
    ActiveCMAES<SelectionPolicyType, TransformationPolicyType>(
        lambda, transformationPolicy, batchSize, maxIterations,
        tolerance, selectionPolicy, stepSize),
    populationFactor(populationFactor),
    maxRestarts(maxRestarts),
    maxFunctionEvaluations(maxFunctionEvaluations)
{ /* Nothing to do. */ }

template<typename SelectionPolicyType,
         typename TransformationPolicyType,
         bool UseBIPOPFlag>
template<typename SeparableFunctionType, typename MatType, typename... CallbackTypes>
typename MatType::elem_type aPOP_CMAES<SelectionPolicyType,
                                       TransformationPolicyType, UseBIPOPFlag>::Optimize(
    SeparableFunctionType& function,
    MatType& iterateIn,
    CallbackTypes&&... callbacks)
{
    // Convenience typedefs.
    typedef typename MatType::elem_type ElemType;

    StoreBestCoordinates<MatType> sbc;
    StoreBestCoordinates<MatType> overallSBC;
    size_t totalFunctionEvaluations = 0;
    size_t largePopulationBudget = 0;
    size_t smallPopulationBudget = 0;

    // First single run with default population size
    MatType iterate = iterateIn;
    ElemType overallObjective = ActiveCMAES<SelectionPolicyType,
                                            TransformationPolicyType>::Optimize(function, iterate, sbc,
                                                                                callbacks...);

    overallSBC = sbc;
    ElemType objective;
    size_t evaluations;

    size_t defaultLambda = this->PopulationSize();
    size_t currentLargeLambda = defaultLambda;

    double stepSizeDefault = this->StepSize();

    // Print out the default population size
    Info << "Default population size: " << defaultLambda << "." << std::endl;

    size_t restart = 0;

    while (restart < maxRestarts)
    {
        if (!UseBIPOPFlag || largePopulationBudget <= smallPopulationBudget ||
            restart == 0 || restart == maxRestarts - 1)
        {
            // Large population regime (IPOP or BIPOP)
            currentLargeLambda *= populationFactor;
            this->PopulationSize() = currentLargeLambda;
            this->StepSize() = stepSizeDefault;

            Info << "POP-CMA-ES: restart " << restart << ", large population size" <<
                " (lambda): " << this->PopulationSize() << "." << std::endl;

            iterate = iterateIn;

            // Optimize using the CMAES object.
            objective = ActiveCMAES<SelectionPolicyType,
                                    TransformationPolicyType>::Optimize(function, iterate, sbc,
                                                                        callbacks...);

            evaluations = this->FunctionEvaluations();
            largePopulationBudget += evaluations;
        }
        else if (UseBIPOPFlag)
        {
            // Small population regime (BIPOP only)
            double u = arma::randu<double>();
            size_t smallLambda = static_cast<size_t>(defaultLambda * std::pow(0.5 *
                                                                                  currentLargeLambda / defaultLambda, u * u));
            double stepSizeSmall = 2 * std::pow(10, -2 * arma::randu<double>());

            this->PopulationSize() = smallLambda;
            this->StepSize() = stepSizeSmall;

            Info << "BIPOP-CMA-ES: restart " << restart << ", small population" <<
                " size (lambda): " << this->PopulationSize() << "." << std::endl;

            iterate = iterateIn;

            // Optimize using the CMAES object.
            objective = ActiveCMAES<SelectionPolicyType,
                                    TransformationPolicyType>::Optimize(function, iterate, sbc,
                                                                        callbacks...);

            evaluations = this->FunctionEvaluations();
            smallPopulationBudget += evaluations;
        }

        if (objective < overallObjective)
        {
            overallObjective = objective;
            overallSBC = sbc;
            Info << "POP-CMA-ES: New best objective: " << overallObjective
                 << "." << std::endl;
        }

        totalFunctionEvaluations += evaluations;
        // Check if the total number of evaluations has exceeded the limit
        if (totalFunctionEvaluations >= maxFunctionEvaluations) {
            Warn << "POP-CMA-ES: Maximum function overall evaluations reached. "
                 << "terminating optimization." << std::endl;

            Callback::EndOptimization(*this, function, iterate, callbacks...);
            iterateIn = std::move(overallSBC.BestCoordinates());
            return overallSBC.BestObjective();
        }

        ++restart;
    }

    Callback::EndOptimization(*this, function, iterate, callbacks...);
    iterateIn = std::move(overallSBC.BestCoordinates());
    return overallSBC.BestObjective();
}

// Define aIPOP_CMAES and aBIPOP_CMAES using the POP_CMAES template
template<typename SelectionPolicyType = FullSelection,
         typename TransformationPolicyType = EmptyTransformation<>>
using aIPOP_CMAES = aPOP_CMAES<SelectionPolicyType, TransformationPolicyType, false>;

template<typename SelectionPolicyType = FullSelection,
         typename TransformationPolicyType = EmptyTransformation<>>
using aBIPOP_CMAES = aPOP_CMAES<SelectionPolicyType, TransformationPolicyType, true>;

}

#endif // APOP_CMAES_HPP
