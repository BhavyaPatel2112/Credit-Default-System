// Plain-language wording for each reason group returned by the service
// (api/features.py REASON_GROUPS). `against` is used on a referral notice,
// `for` when the same group lowered the risk.

const TEXT: Record<string, { against: string; for: string }> = {
  "External credit scores": {
    against: "Scores from external credit agencies are lower than for most approved applicants.",
    for: "Scores from external credit agencies are strong.",
  },
  "Credit bureau record (other lenders)": {
    against: "Borrowing with other lenders is high relative to the credit available.",
    for: "Borrowing with other lenders is well within limits.",
  },
  "Credit bureau record (other lenders): none on file": {
    against: "There is no credit history with other lenders to support the application.",
    for: "Having no borrowing with other lenders counts in favour.",
  },
  "Past instalment payments": {
    against: "Some earlier instalments were paid late or short.",
    for: "Earlier instalments were paid on time.",
  },
  "Past instalment payments: none on file": {
    against: "There is no record of earlier instalments with this lender.",
    for: "No earlier instalments on file.",
  },
  "Previous applications with this lender": {
    against: "Earlier applications with this lender were often refused or cancelled.",
    for: "Earlier applications with this lender went well.",
  },
  "Previous applications with this lender: none on file": {
    against: "This is a first application with this lender.",
    for: "A first application with this lender.",
  },
  "Past point-of-sale and cash loans": {
    against: "Earlier point-of-sale or cash loans show missed payments.",
    for: "Earlier point-of-sale or cash loans were repaid as agreed.",
  },
  "Credit card usage": {
    against: "Credit card balances are high compared with their limits.",
    for: "Credit card use is moderate.",
  },
  "Recent credit enquiries": {
    against: "There have been several recent credit enquiries.",
    for: "Few recent credit enquiries.",
  },
  "Loan and instalment relative to income": {
    against: "The loan and its instalment are large compared with income.",
    for: "The instalment is affordable on the stated income.",
  },
  "Loan amount and repayment term": {
    against: "The loan amount or its repayment term adds risk.",
    for: "The loan amount and term are moderate.",
  },
  Employment: {
    against: "Time in current employment is short, or the employment type carries more risk.",
    for: "Employment is established.",
  },
  Education: { against: "Education level is associated with higher default in past data.", for: "Education level counts in favour." },
  "Car ownership": { against: "Car ownership details add risk.", for: "Car ownership counts in favour." },
  Housing: { against: "Housing situation adds risk.", for: "Housing situation counts in favour." },
  "Region and address": {
    against: "The region of residence, or a recent change of registered address, carries more risk.",
    for: "The region of residence and a settled address count in favour.",
  },
  Household: { against: "Household size adds risk.", for: "Household size counts in favour." },
};
// Release 1.3 removed the time of application, contact details, document
// flags and social-circle defaults from the model (no lender could defend
// them as reasons), so those reason groups no longer occur.

export function plain(reason: string, effect: "raises risk" | "lowers risk") {
  const t = TEXT[reason] ?? TEXT[reason.replace(/: none on file$/, "")];
  if (!t) return reason;
  return effect === "raises risk" ? t.against : t.for;
}

// Reasons an applicant can do nothing about are flagged for the officer.
export const NOT_ACTIONABLE = new Set(["Region and address", "Education"]);
