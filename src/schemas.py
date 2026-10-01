from typing import Optional

from pydantic import BaseModel, Field


class BorrowerInput(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(ge=0)
    age: float = Field(gt=0)

    NumberOfTime30_59DaysPastDueNotWorse: int = Field(ge=0)

    DebtRatio: float = Field(ge=0)
    MonthlyIncome: Optional[float] = Field(default=None, ge=0)

    NumberOfOpenCreditLinesAndLoans: int = Field(ge=0)

    NumberOfTimes90DaysLate: int = Field(ge=0)

    NumberRealEstateLoansOrLines: int = Field(ge=0)

    NumberOfTime60_89DaysPastDueNotWorse: int = Field(ge=0)

    NumberOfDependents: Optional[float] = Field(default=None, ge=0)